#!/usr/bin/env bash
# Kaggle TPU setup for ai-toolkit (tpu-kaggle-support branch).
# Run once per Kaggle TPU session BEFORE training.
set -euo pipefail

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export ACCELERATE_USE_TPU="${ACCELERATE_USE_TPU:-true}"

echo "== torch version =="
python -c "import torch; print(torch.__version__)"

TORCH_VER="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
echo "== installing torch_xla==${TORCH_VER} =="
pip install -q "torch_xla==${TORCH_VER}" -f https://storage.googleapis.com/libtpu-releases/index.html || {
  echo "Pinned torch_xla==${TORCH_VER} failed. Check https://github.com/pytorch/xla/releases"
  echo "and install the matching wheel manually."
  exit 1
}

echo "== TPU visibility check =="
python -c "import torch_xla.core.xla_model as xm; print('devices:', xm.get_xla_supported_devices()); print('hw:', xm.xla_device_hw(xm.xla_device()))"

echo "== accelerate device check =="
python -c "from accelerate import Accelerator; print('accelerator device:', Accelerator().device)"

echo "Done. Run training with: python run.py config/examples/train_tpu_flux2_klein_4b.yaml"
echo "See docs/TPU.md for config rules."
