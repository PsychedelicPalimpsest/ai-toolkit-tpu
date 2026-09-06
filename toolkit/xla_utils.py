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
