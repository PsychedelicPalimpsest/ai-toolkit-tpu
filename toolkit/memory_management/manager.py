import ctypes
import gc

import torch
from .manager_modules import (
    LinearLayerMemoryManager,
    ConvLayerMemoryManager,
    OstrisLinearLayerMemoryManager,
    EmbeddingLayerMemoryManager,
    _DEVICE_STATE,
)
import random

LINEAR_MODULES = [
    "Linear",
    "LoRACompatibleLinear",
    "QLinear",
    'OstrisLinear',
]
CONV_MODULES = [
    "Conv2d",
    "LoRACompatibleConv",
    "QConv2d",
]

UNMANAGED_MODULES = [
    "LayerNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "Embedding",
    "EmbeddingBag",
    "RNNBase",
    "LSTM",
    "GRU",
    "RNN",
    "Conv3d"
]

UNMANAGED_MODULES_INCLUDES = ["RotaryEmbedding", "Norm", "RotaryPosEmbed"]


def _move_param_to_device(owner: torch.nn.Module, name: str, device) -> None:
    """Move ``owner._parameters[name]`` to ``device``.

    Plain ``p.data = p.data.to(device)`` raises
    ``RuntimeError: ... incompatible tensor type`` on cross-backend moves
    (CPU<->XLA, CPU<->meta): ``set_data`` requires identical tensor types.
    ``nn.Module._apply`` (i.e. ``module.to()``) handles this by replacing the
    Parameter object, so mirror that here. No-op when already on device.
    """
    param = owner._parameters.get(name, None)
    if param is None:
        return
    try:
        if param.device == torch.device(device):
            return
    except Exception:
        pass
    moved = param.data.to(device)
    try:
        param.data = moved
    except RuntimeError:
        owner._parameters[name] = torch.nn.Parameter(moved, requires_grad=param.requires_grad)


class MemoryManager:
    def __init__(
        self,
        module: torch.nn.Module,
        process_device: torch.device = torch.device("cpu"),
    ):
        self.module: torch.nn.Module = module
        self.process_device: torch.device = process_device
        self.unmanaged_modules: list[torch.nn.Module] = []

    def memory_managed_to(self, *args, **kwargs):
        # the manager owns placement: the resident (unmanaged/ignore) set must
        # live on the compute device for forwards to work. Legacy parking
        # gestures (.to("cpu") between phases) would strand it there — the
        # swapped .device property keeps reporting the compute device, so no
        # holder heal ever brings it back. Honor device moves only TO the
        # compute device; skip the device part of anything else (dtype
        # handling below is unaffected).
        target_device = kwargs.get("device", None)
        for arg in args:
            if isinstance(arg, (torch.device, str)) and not isinstance(arg, torch.dtype):
                try:
                    target_device = torch.device(arg)
                except (TypeError, RuntimeError):
                    pass
            elif isinstance(arg, torch.device):
                target_device = arg
        move_resident = target_device is not None and (
            torch.device(target_device) == torch.device(self.process_device)
        )
        if target_device is None or move_resident:
            # first move all the unmanaged modules
            for module in self.unmanaged_modules:
                if isinstance(module, torch.Tensor):
                    # Parameters and bare tensor buffers cannot move this way
                    module.data = module.data.to(*args, **kwargs)
                else:
                    module.to(*args, **kwargs)
        # check for a dtype argument
        dtype = None
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
        elif len(args) > 0:
            for i, arg in enumerate(args):
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break
        if dtype is not None:
            return self.module._mm_to(dtype=dtype)
        return self.module

    @classmethod
    def attach(
        cls,
        module: torch.nn.Module,
        device: torch.device,
        offload_percent: float = 1.0,
        ignore_modules: list[torch.nn.Module] = []
    ):
        if hasattr(module, "_memory_manager"):
            # already attached
            return

        # TPU/XLA: the bouncing offloader (pinned-CPU weights +
        # torch.cuda.Stream/Event staging) has no XLA equivalent, and manual
        # `p.data = p.data.to(xla)` raises "incompatible tensor type"
        # (only nn.Module.to() handles cross-backend moves). Place the model
        # plainly and skip all layer managers; offload fractions are ignored.
        try:
            _attach_device_type = torch.device(device).type
        except Exception:
            _attach_device_type = None
        if _attach_device_type == "xla":
            try:
                from toolkit.xla_utils import warn_once
                warn_once(
                    "MemoryManager layer offloading is CUDA-only and disabled on "
                    "TPU/XLA; placing model directly on the XLA device."
                )
            except Exception:
                pass
            module.to(device)
            return

        module._memory_manager = cls(module, device)

        # override the to method to handle memory management
        module._mm_to = module.to
        module.to = module._memory_manager.memory_managed_to

        # a fully offloaded module's parameters all live on cpu, which makes
        # ModelMixin.device (and pipelines deriving their execution device
        # from it) report "cpu" and plant latents/timesteps there. Report the
        # compute device instead via an in-place subclass (same class-swap
        # pattern as OstrisLinear/adopt_component); detach() restores it.
        try:
            module._mm_orig_class = module.__class__
            managed_cls = type(
                module.__class__.__name__,
                (module.__class__,),
                {
                    "device": property(
                        lambda self: self._memory_manager.process_device,
                        lambda self, value: self.__dict__.__setitem__(
                            "_mm_device_shadow", value
                        ),
                    )
                },
            )
            # transformers keys per-class registries (e.g. the hidden-states
            # capture specs) by str(model.__class__); make the subclass
            # stringify identically so those lookups still hit
            managed_cls.__module__ = module.__class__.__module__
            managed_cls.__qualname__ = module.__class__.__qualname__
            module.__class__ = managed_cls
        except TypeError:
            # exotic class layouts (__slots__ etc.): keep the original class
            module._mm_orig_class = None

        # add ignore modules to unmanaged list; they must stay RESIDENT on the
        # compute device (fp32 tables, pad tokens) — a model attached while
        # parked on cpu would otherwise feed cpu tensors into gpu math
        for im in ignore_modules:
            module._memory_manager.unmanaged_modules.append(im)
            try:
                if isinstance(im, torch.Tensor):
                    im.data = im.data.to(device)
                elif isinstance(im, torch.nn.Module):
                    im.to(device)
            except Exception:
                pass

        # count ignore modules as processed — including their whole subtree: an
        # ignored module stays resident, so none of its children may get a
        # layer manager (a managed child would pin its weight back to cpu
        # behind the parent's back)
        modules_processed = [x for x in ignore_modules]
        for im in ignore_modules:
            if isinstance(im, torch.nn.Module):
                modules_processed.extend(im.modules())

        # weights tied to an embedding (lm_head <-> embed_tokens) must not be
        # managed: pinning the linear's weight to cpu strands the (unmanaged)
        # embedding that shares the same tensor
        embedding_weight_ptrs = {
            m.weight.data_ptr()
            for m in module.modules()
            if isinstance(m, torch.nn.Embedding)
        }
        # weights of embeddings that get MANAGED (cpu-resident bouncing): a
        # linear sharing one of these must be managed too, not left resident
        managed_embedding_ptrs = set()
        # attach to all modules
        for name, sub_module in module.named_modules():
            for child_name, child_module in sub_module.named_modules():
                if (
                    child_module.__class__.__name__ in LINEAR_MODULES
                    and child_module not in modules_processed
                ):
                    skip = False
                    if offload_percent < 1.0:
                        # randomly skip some modules
                        if random.random() > offload_percent:
                            skip = True
                    if (
                        not getattr(child_module, "is_ostris_quantized", False)
                        and isinstance(getattr(child_module, "weight", None), torch.Tensor)
                        and child_module.weight.data_ptr() in embedding_weight_ptrs
                        and child_module.weight.data_ptr()
                        not in managed_embedding_ptrs
                    ):
                        skip = True
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                    else:
                        # linear; OstrisLinear bounces its quantized buffers instead
                        # of a dequantized weight (module.weight is a property)
                        if getattr(child_module, "is_ostris_quantized", False):
                            OstrisLinearLayerMemoryManager.attach(
                                child_module, module._memory_manager
                            )
                        else:
                            LinearLayerMemoryManager.attach(
                                child_module, module._memory_manager
                            )
                        # attach to ARA as well
                        if hasattr(child_module, "ara_lora_ref"):
                            ara = child_module.ara_lora_ref()
                            if ara not in modules_processed:
                                MemoryManager.attach(
                                    ara,
                                    device,
                                )
                    modules_processed.append(child_module)
                elif (
                    child_module.__class__.__name__ in CONV_MODULES
                    and child_module not in modules_processed
                ):
                    skip = False
                    if offload_percent < 1.0:
                        # randomly skip some modules
                        if random.random() > offload_percent:
                            skip = True
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                    else:
                        # conv
                        ConvLayerMemoryManager.attach(
                            child_module, module._memory_manager
                        )
                        # attach to ARA as well
                        if hasattr(child_module, "ara_lora_ref"):
                            ara = child_module.ara_lora_ref()
                            if ara not in modules_processed:
                                MemoryManager.attach(
                                    ara,
                                    device,
                                )
                            modules_processed.append(ara)
                    modules_processed.append(child_module)
                elif (
                    isinstance(child_module, torch.nn.Embedding)
                    and child_module not in modules_processed
                    and child_module.weight.numel()
                    * child_module.weight.element_size()
                    > 64 * 1024 * 1024
                ):
                    # (the cpu gather is autograd-transparent, so a trainable
                    # embedding still gets grads — they just land on cpu)
                    # large frozen vocab table: cpu-resident, rows gathered on
                    # cpu (only the looked-up tokens cross the bus)
                    EmbeddingLayerMemoryManager.attach(
                        child_module, module._memory_manager
                    )
                    # a tied lm_head must bounce the (now cpu-resident) shared
                    # weight rather than stay resident; record both the pre-
                    # and post-move ptrs (a cpu->cpu move keeps the tensor)
                    managed_embedding_ptrs.add(child_module.weight.data_ptr())
                    embedding_weight_ptrs.add(child_module.weight.data_ptr())
                    modules_processed.append(child_module)
                elif child_module.__class__.__name__ in UNMANAGED_MODULES or any(
                    inc in child_module.__class__.__name__
                    for inc in UNMANAGED_MODULES_INCLUDES
                ):
                    # unmanaged — but never re-list a module the nested walk
                    # already managed (a managed Embedding landing here would
                    # get hauled back to the gpu by memory_managed_to), and
                    # don't append duplicates
                    if (
                        child_module not in modules_processed
                        and not hasattr(child_module, "_layer_memory_manager")
                        and child_module
                        not in module._memory_manager.unmanaged_modules
                    ):
                        module._memory_manager.unmanaged_modules.append(child_module)
                else:
                    continue

        # everything NOT managed is the resident set and must live on the
        # compute device. A model attached while parked on cpu (the offload
        # load flow) otherwise keeps its rotary buffers / norms / conv towers
        # on cpu and the first forward explodes on a device mismatch. Managed
        # layers (pinned-cpu weights, cpu-resident bouncing embeddings) are
        # skipped via their _layer_memory_manager.
        for sub in module.modules():
            if hasattr(sub, "_layer_memory_manager"):
                continue
            for pname, p in list(sub._parameters.items()):
                if p is not None:
                    # safe cross-backend move (plain `p.data = p.data.to()`
                    # raises "incompatible tensor type" on CPU<->XLA/meta)
                    _move_param_to_device(sub, pname, device)
            for name, b in sub._buffers.items():
                if b is not None and b.device != device:
                    sub._buffers[name] = b.to(device)

    @classmethod
    def detach(cls, module: torch.nn.Module):
        """
        Reverse of attach(). Moves unmanaged modules back to CPU, restores the
        original .to() and forward methods on all child layers, unpins CPU weight
        tensors, and clears the global CUDA device state.

        Call this before unloading/replacing a module that had attach() applied.
        """
        if not hasattr(module, "_memory_manager"):
            return

        if getattr(module, "_mm_orig_class", None) is not None:
            module.__class__ = module._mm_orig_class
            del module._mm_orig_class

        for unmanaged in module._memory_manager.unmanaged_modules:
            try:
                if isinstance(unmanaged, torch.Tensor):
                    unmanaged.data = unmanaged.data.to('cpu')
                else:
                    unmanaged.to('cpu')
            except Exception:
                pass

        if hasattr(module, "_mm_to"):
            module.to = module._mm_to
            del module._mm_to

        del module._memory_manager

        for child in module.modules():
            lmm = getattr(child, "_layer_memory_manager", None)
            if lmm is None:
                continue

            original_forward = getattr(lmm, "_original_forward", None)
            if original_forward is not None:
                if hasattr(child, "ara_lora_ref"):
                    ara = child.ara_lora_ref()
                    if ara is not None:
                        ara.org_forward = original_forward
                else:
                    child.forward = original_forward

            for param_name in ("weight", "bias"):
                # read _parameters directly: OstrisLinear.weight is a property that
                # materializes a full dequantized weight on access
                param = child._parameters.get(param_name, None)
                if param is None or not isinstance(param, torch.nn.Parameter):
                    continue
                try:
                    if param.data.is_pinned():
                        object.__setattr__(
                            child,
                            param_name,
                            torch.nn.Parameter(
                                param.data.clone(),
                                requires_grad=param.requires_grad,
                            ),
                        )
                except Exception:
                    pass

            if getattr(child, "is_ostris_quantized", False):
                # move quantized buffers home and unpin them (clone drops pinning)
                for buf_name, buf in list(child._buffers.items()):
                    if buf is None:
                        continue
                    try:
                        if buf.device.type != "cpu":
                            buf = buf.to("cpu")
                        if buf.is_pinned():
                            buf = buf.clone()
                        child._buffers[buf_name] = buf
                    except Exception:
                        pass

            del child._layer_memory_manager
            if hasattr(child, "_memory_management_device"):
                del child._memory_management_device
            if hasattr(child, "_is_memory_managed"):
                del child._is_memory_managed

        keys_to_delete = [
            dev for dev in _DEVICE_STATE
            if isinstance(dev, torch.device) and dev.type == "cuda"
        ]
        for key in keys_to_delete:
            del _DEVICE_STATE[key]

        try:
            from toolkit.xla_utils import empty_cache
            empty_cache()
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @classmethod
    def free(cls, module: torch.nn.Module):
        """
        Detach memory management (if attached) and destroy the module's weights
        by moving them to the meta device.

        Unlike detach(), nothing is staged back to CPU first: to('meta') frees
        each storage from wherever it currently lives, so no transient host
        allocation is made for data that is about to be discarded, and pinned
        tensors are freed without the clone that unpinning requires. Freed
        pinned storages land in torch's caching host allocator, not the OS;
        call release_cached_memory() afterward to get the RSS back.
        """
        if hasattr(module, "_memory_manager"):
            if hasattr(module, "_mm_to"):
                module.to = module._mm_to
                del module._mm_to

            del module._memory_manager

            for child in module.modules():
                lmm = getattr(child, "_layer_memory_manager", None)
                if lmm is None:
                    continue

                original_forward = getattr(lmm, "_original_forward", None)
                if original_forward is not None:
                    if hasattr(child, "ara_lora_ref"):
                        ara = child.ara_lora_ref()
                        if ara is not None:
                            ara.org_forward = original_forward
                    else:
                        child.forward = original_forward

                del child._layer_memory_manager
                if hasattr(child, "_memory_management_device"):
                    del child._memory_management_device
                if hasattr(child, "_is_memory_managed"):
                    del child._is_memory_managed

            keys_to_delete = [
                dev for dev in _DEVICE_STATE
                if isinstance(dev, torch.device) and dev.type == "cuda"
            ]
            for key in keys_to_delete:
                del _DEVICE_STATE[key]

        # bypass any overridden/nopped-out .to() so the storages are actually freed
        torch.nn.Module.to(module, "meta")
        try:
            from toolkit.xla_utils import empty_cache
            empty_cache()
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @classmethod
    def release_cached_memory(cls):
        """
        Return freed memory to the OS. Freed pinned-host storages sit in
        torch's caching host allocator and freed pageable memory sits in
        glibc's arenas; neither shows up as reclaimed RSS without an
        explicit flush. Call after free()ing a large module.
        """
        gc.collect()
        try:
            from toolkit.xla_utils import empty_cache
            empty_cache()
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # torch's pinned-host cache; private API, name varies by torch version
        for fn_name in ("_accelerator_emptyHostCache", "_host_emptyCache"):
            fn = getattr(torch._C, fn_name, None)
            if fn is not None:
                try:
                    fn()
                    break
                except Exception:
                    pass

        # glibc keeps freed arenas mapped; CDLL(None) resolves malloc_trim in
        # the running process where glibc is present and fails cleanly on
        # macOS/musl
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except Exception:
            pass
