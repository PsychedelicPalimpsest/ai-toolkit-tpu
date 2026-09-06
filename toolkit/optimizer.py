import torch


# bitsandbytes 8-bit optimizers are CUDA-only. On TPU/XLA (or when bnb fails to
# import) fall back to the equivalent torch optimizer so configs like
# `optimizer: adamw8bit` keep working. See docs/TPU.md.
_BNB_FALLBACK = {
    "adam8bit": ("adam", torch.optim.Adam),
    "adamw8bit": ("adamw", torch.optim.AdamW),
    "lion8bit": ("lion", None),
    "ademamix8bit": ("adamw", torch.optim.AdamW),
}


def _params_on_xla(params) -> bool:
    try:
        from toolkit.xla_utils import is_xla_device
        for group in params if isinstance(params, list) and params and isinstance(params[0], dict) else [{"params": params}]:
            for p in group["params"]:
                if isinstance(p, torch.Tensor) and is_xla_device(p.device):
                    return True
                # params may not be materialized yet; check torch_xla presence + env
                if isinstance(p, torch.Tensor) and str(p.device).startswith("xla"):
                    return True
    except Exception:
        pass
    try:
        from toolkit.xla_utils import is_xla_available, has_tpu
        # If params live on CPU during init but we are on a TPU host, still avoid bnb:
        # bnb import itself fails without CUDA libs on Kaggle TPU VMs.
        if is_xla_available() and has_tpu():
            return True
    except Exception:
        pass
    return False


def get_optimizer(
        params,
        optimizer_type='adam',
        learning_rate=1e-6,
        optimizer_params=None
):
    if optimizer_params is None:
        optimizer_params = {}
    lower_type = optimizer_type.lower()
    if lower_type.startswith("dadaptation"):
        # dadaptation optimizer does not use standard learning rate. 1 is the default value
        import dadaptation
        print("Using DAdaptAdam optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0
        if lower_type.endswith('lion'):
            optimizer = dadaptation.DAdaptLion(params, eps=1e-6, lr=use_lr, **optimizer_params)
        elif lower_type.endswith('adam'):
            optimizer = dadaptation.DAdaptLion(params, eps=1e-6, lr=use_lr, **optimizer_params)
        elif lower_type == 'dadaptation':
            # backwards compatibility
            optimizer = dadaptation.DAdaptAdam(params, eps=1e-6, lr=use_lr, **optimizer_params)
            # warn user that dadaptation is deprecated
            print("WARNING: Dadaptation optimizer type has been changed to DadaptationAdam. Please update your config.")
    elif lower_type.startswith("prodigy8bit"):
        from toolkit.optimizers.prodigy_8bit import Prodigy8bit
        print("Using Prodigy optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0

        print(f"Using lr {use_lr}")
        # let net be the neural network you want to train
        # you can choose weight decay value based on your problem, 0 by default
        optimizer = Prodigy8bit(params, lr=use_lr, eps=1e-6, **optimizer_params)
    elif lower_type.startswith("prodigy"):
        from prodigyopt import Prodigy

        print("Using Prodigy optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0

        print(f"Using lr {use_lr}")
        # let net be the neural network you want to train
        # you can choose weight decay value based on your problem, 0 by default
        optimizer = Prodigy(params, lr=use_lr, eps=1e-6, **optimizer_params)
    elif lower_type == "adam8":
        from toolkit.optimizers.adam8bit import Adam8bit

        optimizer = Adam8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
    elif lower_type == "adamw8":
        from toolkit.optimizers.adam8bit import Adam8bit

        optimizer = Adam8bit(params, lr=learning_rate, eps=1e-6, decouple=True, **optimizer_params)
    elif lower_type.endswith("8bit"):
        # TPU/XLA has no bitsandbytes kernels -> fall back to torch equivalent.
        if _params_on_xla(params):
            from toolkit.xla_utils import warn_once
            fallback_name, fallback_cls = _BNB_FALLBACK.get(lower_type, ("adamw", torch.optim.AdamW))
            warn_once(
                f"Optimizer '{optimizer_type}' is CUDA-only (bitsandbytes) and not available on TPU/XLA. "
                f"Falling back to '{fallback_name}'. For TPU use optimizer: {fallback_name} directly "
                f"to silence this warning."
            )
            if fallback_cls is None:  # lion8bit without lion_pytorch
                from lion_pytorch import Lion
                return Lion(params, lr=learning_rate, **optimizer_params)
            if fallback_name == "adam":
                return fallback_cls(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)
            return fallback_cls(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)
        try:
            import bitsandbytes
        except (ImportError, OSError) as e:
            # Kaggle TPU VMs have no CUDA libs: bnb import fails. Fall back
            # instead of crashing so TPU configs keep working.
            from toolkit.xla_utils import warn_once
            fallback_name, fallback_cls = _BNB_FALLBACK.get(lower_type, ("adamw", torch.optim.AdamW))
            warn_once(
                f"bitsandbytes import failed ({e}); falling back from '{optimizer_type}' "
                f"to '{fallback_name}'. Use optimizer: {fallback_name} on TPU/CPU."
            )
            if fallback_cls is None:
                from lion_pytorch import Lion
                return Lion(params, lr=learning_rate, **optimizer_params)
            return fallback_cls(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)

        if lower_type == "adam8bit":
            return bitsandbytes.optim.Adam8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
        if lower_type == "ademamix8bit":
            return bitsandbytes.optim.AdEMAMix8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
        elif lower_type == "adamw8bit":
            return bitsandbytes.optim.AdamW8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
        elif lower_type == "lion8bit":
            return bitsandbytes.optim.Lion8bit(params, lr=learning_rate, **optimizer_params)
        else:
            raise ValueError(f'Unknown optimizer type {optimizer_type}')
    elif lower_type == 'adam':
        optimizer = torch.optim.Adam(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)
    elif lower_type == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)
    elif lower_type == 'lion':
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=learning_rate, **optimizer_params)
        except ImportError:
            raise ImportError("Please install lion_pytorch to use Lion optimizer -> pip install lion-pytorch")
    elif lower_type == 'adagrad':
        optimizer = torch.optim.Adagrad(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'adafactor':
        from toolkit.optimizers.adafactor import Adafactor
        if 'relative_step' not in optimizer_params:
            optimizer_params['relative_step'] = False
        if 'scale_parameter' not in optimizer_params:
            optimizer_params['scale_parameter'] = False
        if 'warmup_init' not in optimizer_params:
            optimizer_params['warmup_init'] = False
        optimizer = Adafactor(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'automagic':
        from toolkit.optimizers.automagic import Automagic
        optimizer = Automagic(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'automagic2':
        from toolkit.optimizers.automagic2 import Automagic2
        optimizer = Automagic2(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'automagic3':
        from toolkit.optimizers.automagic3 import Automagic3
        optimizer = Automagic3(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'automagicexperiment':
        from toolkit.optimizers.automagicEXPERIMENT import AutomagicEXPERIMENT
        optimizer = AutomagicEXPERIMENT(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'adamconvrot':
        from toolkit.optimizers.adamconvrot import AdamConvRot
        if 'eps' not in optimizer_params:
            optimizer_params['eps'] = 1e-6
        optimizer = AdamConvRot(params, lr=float(learning_rate), **optimizer_params)
    else:
        raise ValueError(f'Unknown optimizer type {optimizer_type}')
    return optimizer
