"""TPU / XLA compatibility helpers for ai-toolkit.

Kaggle TPUs expose PyTorch devices as ``xla:0`` .. ``xla:7`` via ``torch_xla``.
The rest of the codebase was written CUDA-first (``torch.cuda.*``,
``bitsandbytes`` 8-bit optimizers, ``torchao``/``quanto`` fp8 quantization,
``torch.cuda.Stream``/``Event`` layer offloading). This module centralizes all
device branching so training can run on:

* CUDA (unchanged behaviour)
* CPU (unchanged behaviour)
* XLA / TPU (graceful fallbacks + warnings)
* MPS (pre-existing handling, untouched)

All helpers are safe to call when ``torch_xla`` is NOT installed — they simply
report ``False`` / no-op. That keeps CUDA-only installs dependency-free.
"""

import gc
import warnings

import torch


_XLA_AVAILABLE = None
_HAS_TPU = None


def is_xla_available() -> bool:
    """True if ``torch_xla`` is importable in this environment."""
    global _XLA_AVAILABLE
    if _XLA_AVAILABLE is None:
        try:
            import torch_xla  # noqa: F401
            _XLA_AVAILABLE = True
        except ImportError:
            _XLA_AVAILABLE = False
    return _XLA_AVAILABLE


def is_xla_device(device) -> bool:
    """True if a torch.device / string refers to an XLA device."""
    if device is None:
        return False
    try:
        dtype = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    except Exception:
        return False
    return dtype == "xla"


def is_cuda_device(device) -> bool:
    if device is None:
        return False
    try:
        dtype = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    except Exception:
        return False
    return dtype == "cuda"


def has_tpu() -> bool:
    """True if a TPU is reachable via torch_xla (Kaggle TPU VM / Colab TPU)."""
    global _HAS_TPU
    if _HAS_TPU is None:
        if not is_xla_available():
            _HAS_TPU = False
        else:
            try:
                import torch_xla.core.xla_model as xm
                _HAS_TPU = xm.xla_device_hw(xm.xla_device()) == "TPU"
            except Exception:
                _HAS_TPU = False
    return _HAS_TPU


def get_xla_device(index: int = 0) -> torch.device:
    """Return ``xla:index`` device. Raises a helpful error if torch_xla missing."""
    if not is_xla_available():
        raise RuntimeError(
            "Requested XLA device but torch_xla is not installed. "
            "On Kaggle TPUs install a torch+torch_xla pinned pair, e.g. "
            "`pip install torch==2.5.0 torch_xla==2.5.0 -f "
            "https://storage.googleapis.com/libtpu-releases/index.html` "
            "(match versions to your torch). See docs/TPU.md."
        )
    return torch.device(f"xla:{index}")


def empty_cache():
    """Device-agnostic cache flush. Safe on CUDA/CPU/XLA/MPS."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    # XLA has no empty_cache; stepping / GC releases HBM-backed live tensors.
    # xm.mark_step() is issued by the training loop, not here.
    gc.collect()


def synchronize(device=None):
    """Device-agnostic synchronize (profiler / benchmark barriers)."""
    try:
        if is_xla_device(device):
            import torch_xla.core.xla_model as xm
            xm.wait_device_ops()
            return
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            if device is not None and is_cuda_device(device):
                with torch.cuda.device(device):
                    torch.cuda.synchronize()
            else:
                torch.cuda.synchronize()
        except Exception:
            pass


def seed_all(seed: int):
    """Seed python/torch/cuda/xla RNGs without crashing on missing backends."""
    import random
    random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    if is_xla_available():
        try:
            import torch_xla.core.xla_model as xm
            xm.set_rng_state(seed)
        except Exception:
            pass


def autocast_device_type(device) -> str:
    """Map a torch device to a valid ``torch.autocast(device_type=...)`` string.

    * cuda -> "cuda"
    * cpu/xla/mps -> "cpu" (XLA autocast kernels are not universally available;
      bf16 training on TPU works via explicit dtypes + accelerator.autocast).
    """
    if is_cuda_device(device):
        return "cuda"
    return "cpu"


def mark_step():
    """Issue an XLA step barrier if on XLA, else no-op.

    Required for TPU execution: XLA is lazy and only runs the graph on
    ``mark_step`` / ``optimizer_step``. Safe to call unconditionally at low
    frequency (end of optimizer step, sampling, saving).
    """
    if not is_xla_available():
        return
    try:
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    except Exception:
        pass


def optimizer_step(optimizer, barrier: bool = False):
    """XLA-aware optimizer step: ``xm.optimizer_step`` on XLA, else plain step.

    ``xm.optimizer_step`` issues the mark_step + distributed all-reduce that a
    plain ``optimizer.step()`` would skip on TPU.
    """
    if is_xla_available():
        try:
            import torch_xla.core.xla_model as xm
            # Only route through XLA if any param actually lives on XLA.
            try:
                devices = {p.device.type for g in optimizer.param_groups for p in g["params"]}
            except Exception:
                devices = set()
            if "xla" in devices:
                xm.optimizer_step(optimizer, barrier=barrier)
                return
        except Exception:
            pass
    optimizer.step()


def is_oom_error(exc: BaseException) -> bool:
    """True for CUDA, CPU-allocator, and XLA HBM OOMs."""
    msg = str(exc)
    return (
        isinstance(exc, torch.cuda.OutOfMemoryError)
        or "out of memory" in msg.lower()
        or "HBM" in msg
        or "XLA" in msg and "memory" in msg.lower()
    )


def warn_once(message: str):
    warnings.warn(f"[ai-toolkit][TPU-compat] {message}", stacklevel=3)


# ---------------------------------------------------------------------------
# Multi-core TPU (data-parallel across TPU cores via xmp.spawn).
#
# Single-process / CUDA / CPU behaviour is unchanged: every helper below
# degrades to ordinal 0 / world size 1 / no-op barrier when torch_xla is
# missing or only one replica is running.
# ---------------------------------------------------------------------------

def get_ordinal() -> int:
    """XLA replica ordinal (0 .. world_size-1). 0 when XLA is unavailable."""
    if not is_xla_available():
        return 0
    try:
        import torch_xla.core.xla_model as xm
        return int(xm.get_ordinal())
    except Exception:
        return 0


def get_world_size() -> int:
    """Number of XLA replicas. 1 when XLA is unavailable or single-core."""
    if not is_xla_available():
        return 1
    try:
        import torch_xla.runtime as xr
        return max(1, int(xr.world_size()))
    except Exception:
        pass
    try:
        import torch_xla.core.xla_model as xm
        return max(1, int(xm.xrt_world_size()))
    except Exception:
        return 1


def is_master_ordinal() -> bool:
    """True on replica 0 (or anywhere when not running multi-core XLA)."""
    return get_ordinal() == 0


def is_xla_multiprocess() -> bool:
    """True when several XLA replicas train together (xmp.spawn, N>1)."""
    return is_xla_available() and get_world_size() > 1


def is_global_main_process(accelerator=None) -> bool:
    """Rank-0 check that stays correct under TPU multi-core spawn.

    Plain ``accelerator.is_main_process`` is True in *every* spawned XLA
    worker (each builds its own ``Accelerator`` with world_size 1), so disk
    IO guarded only by it would run N times. This additionally requires XLA
    ordinal 0. Outside multi-core XLA it is exactly ``is_main_process``.
    """
    if accelerator is not None:
        try:
            if not accelerator.is_main_process:
                return False
        except Exception:
            pass
    return is_master_ordinal()


def rendezvous(tag: str):
    """Cross-replica barrier under xmp.spawn; no-op otherwise.

    Use next to ``accelerator.wait_for_everyone()`` at save/sample/cache
    points so workers do not race ahead while rank 0 writes checkpoints.
    """
    if not is_xla_multiprocess():
        return
    try:
        import torch_xla.core.xla_model as xm
        xm.rendezvous(tag)
    except Exception:
        pass


def reduce_mean_scalar(value: float) -> float:
    """Mean of a python scalar across XLA replicas (logging only).

    The training loss is already backpropagated locally; this just keeps
    logged/saved loss values representative of the global batch.
    """
    if not is_xla_multiprocess():
        return float(value)
    try:
        import torch_xla.core.xla_model as xm
        t = torch.tensor(float(value), device=xm.xla_device())
        mean = xm.all_reduce(xm.REDUCE_SUM, t) / get_world_size()
        mark_step()
        return float(mean.detach().to("cpu"))
    except Exception:
        return float(value)


def shard_for_tpu_rank(items, sort: bool = True):
    """Shard a file/item list across XLA replicas (DistributedSampler-style).

    Each rank keeps ``items[ordinal::world_size]`` so every epoch the union
    of ranks covers the dataset once. Single-process: returns ``items``
    untouched (no sorting either) so CUDA/CPU behaviour is bit-identical.
    """
    if not is_xla_multiprocess():
        return items
    items = list(items)
    if sort:
        try:
            items = sorted(items)
        except Exception:
            pass
    return items[get_ordinal()::get_world_size()]


def reseed_for_tpu_rank(base_seed=None):
    """Give each TPU replica a different shuffle/noise RNG stream.

    Call AFTER model/weight init (which must stay identical on all ranks)
    but BEFORE dataloader iteration. No-op unless multi-core XLA.
    Returns the agreed base seed (or None when single-process).
    """
    if not is_xla_multiprocess():
        return base_seed
    import random
    if base_seed is None:
        draw = str(random.SystemRandom().randint(0, 2 ** 31 - 1))
        try:
            import torch_xla.core.xla_model as xm
            # rendezvous returns every replica's payload; [0] is rank 0's,
            # so all ranks agree on one base seed.
            base_seed = int(xm.rendezvous("aitk_base_seed", draw)[0])
        except Exception:
            base_seed = int(draw)
    seed_all(int(base_seed) + get_ordinal())
    return int(base_seed)
