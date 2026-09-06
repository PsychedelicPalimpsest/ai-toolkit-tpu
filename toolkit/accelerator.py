from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module

global_accelerator = None


def get_accelerator() -> Accelerator:
    global global_accelerator
    if global_accelerator is None:
        global_accelerator = Accelerator()
        # TPU visibility check: on Kaggle TPU VMs accelerate only returns an
        # xla device when torch_xla is installed and the TPU is reachable.
        # Warn early instead of silently training on CPU.
        try:
            from toolkit.xla_utils import is_xla_available, has_tpu
            if is_xla_available() and has_tpu() and str(global_accelerator.device).split(":")[0] != "xla":
                print(
                    "[ai-toolkit][TPU-compat] TPU detected via torch_xla but "
                    f"Accelerator chose device '{global_accelerator.device}'. "
                    "If you expected xla, ensure torch_xla matches your torch "
                    "version and that PJRT_DEVICE=TPU is set (see docs/TPU.md)."
                )
            elif str(global_accelerator.device).split(":")[0] == "xla":
                print(f"[ai-toolkit][TPU-compat] Using XLA device {global_accelerator.device}")
        except Exception:
            pass
    return global_accelerator

def unwrap_model(model):
    try:
        accelerator = get_accelerator()
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
    except Exception as e:
        pass
    return model
