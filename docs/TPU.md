# Training on Google Kaggle TPUs (TPU-compat fork)

This branch (`tpu-kaggle-support`) adds graceful TPU/XLA support to ai-toolkit.
CUDA remains the primary path; on a TPU host the same configs run with
automatic fallbacks instead of `torch.cuda.*` / `bitsandbytes` crashes.

## Fixed: `MemoryManager.attach` crash on TPU

Stock ai-toolkit crashes during `sd.load_model()` on Kaggle TPUs with:

```
File "toolkit/memory_management/manager.py", line 295, in attach
    p.data = p.data.to(device)
RuntimeError: Attempted to call `variable.set_data(tensor)`, but `variable`
and `tensor` have incompatible tensor type.
```

Root cause: the layer offloader moves the resident parameter set with manual
`p.data = p.data.to(device)`. `set_data` requires identical tensor types, so
any cross-backend move (CPU→XLA, CPU→meta) raises — while `nn.Module.to()`
handles it by replacing the Parameter object. This fork fixes it three ways:

1. `MemoryManager.attach()` refuses XLA targets outright: it warns once,
   places the model with plain `module.to(device)`, and installs no bouncing
   layers (CUDA-stream offloading has no XLA equivalent).
2. `aitk_post_load` (`toolkit/models/v2/_mixin.py`) and
   `component_load_kwargs` (`toolkit/models/base_model.py`) force
   `offload = 0.0` for XLA devices even if the config still requests it.
3. Every remaining manual `p.data = p.data.to(...)` move
   (`manager.py`, `manager_modules.py`) falls back to Parameter replacement
   on `RuntimeError`, mirroring `nn.Module._apply`.

Verified without a TPU by using the `meta` device, which triggers the
identical `set_data` type check: the old line reproduces the exact error,
the helper moves all params cleanly, and CPU attach/detach behaviour is
byte-for-byte unchanged.

## What changed

Central helper: `toolkit/xla_utils.py`

* `is_xla_available()` / `has_tpu()` / `is_xla_device()` — safe when
  `torch_xla` is NOT installed (returns `False`, no new dependency).
* `empty_cache()` / `synchronize()` / `seed_all()` — drop-in replacements for
  `torch.cuda.*` calls.
* `autocast_device_type()` — `"cuda"` on GPU, `"cpu"` elsewhere (XLA has no
  CUDA autocast kernels).
* `optimizer_step()` / `mark_step()` — `xm.optimizer_step` + `mark_step` on
  XLA (required: XLA is lazy), plain `optimizer.step()` elsewhere.
* `is_oom_error()` — matches CUDA, CPU-allocator, and XLA HBM OOMs.

Behavioural fallbacks (all warn once, never silent):

| Your config | On CUDA | On TPU/XLA host |
|---|---|---|
| `optimizer: adamw8bit` (or any `*8bit`, `adam8`, `adamw8`) | `bitsandbytes` | `adamw`/`adam` torch fallback (`toolkit/optimizer.py`) |
| `model.quantize / quantize_te` (`qfloat8`/`float8`) | torchao/quanto | disabled, full precision (`config_modules.py`, `util/quantize.py`) |
| `model.low_vram / layer_offloading` | `torch.cuda.Stream/Event` offload | disabled, model stays on `xla` (`config_modules.py`, `memory_management/`) |
| `train.xformers / sdp / attention_backend: flash` | CUDA kernels | forced off / `native` (`config_modules.py`) |
| `train.dtype: bf16` | autocast bf16 | bf16 (correct TPU dtype; fp16 warns) |
| `torch.cuda.*` RNG / cache / sync | as before | `seed_all` / no-op cache / `xm.wait_device_ops` |
| `add_model_gpu_splitter_to_flux` | splits over `cuda:N` | no-op (avoids `total/0` crash) |

Training loop (`extensions_built_in/sd_trainer/SDTrainer.py`,
`jobs/process/BaseSDTrainProcess.py`) already used
`accelerator.prepare / backward / accumulate / clip_grad_norm_` correctly, so
no rewrite was needed — only the optimizer step is now XLA-aware and OOM
handling matches XLA messages.

## Kaggle TPU setup

Kaggle provides TPUv3-8 VMs. You need a **pinned torch + torch_xla pair**
matching your torch version, plus `PJRT_DEVICE=TPU`.

```bash
# 1. In Kaggle notebook settings: Accelerator = TPU
# 2. Check versions, then install the matching libtpu wheel:
pip show torch | grep Version
# e.g. torch 2.5.0 -> torch_xla 2.5.0
pip install "torch_xla==2.5.0" -f https://storage.googleapis.com/libtpu-releases/index.html

# 3. Env (do this before `import torch` in every session):
export PJRT_DEVICE=TPU
export XLA_USE_SPMD=1          # optional, better TPU utilization
export ACCELERATE_USE_TPU=true # lets accelerate pick xla device

# 4. Verify TPU is visible:
python -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices()); print(xm.xla_device_hw(xm.xla_device()))"
# expect: ['TPU:0' ... 'TPU:7'] / TPU

# 5. Install ai-toolkit WITHOUT bitsandbytes (CUDA-only, import fails on TPU):
# edit requirements_base.txt or:
pip install -r requirements_base.txt --ignore-installed 2>&1 | grep -v bitsandbytes || true
pip install -e . --no-deps
# then install everything except bitsandbytes manually if needed.
```

Or run `scripts/setup_kaggle_tpu.sh` (installs torch_xla + prints device check).

## TPU config rules

See `config/examples/train_tpu_flux2_klein_4b.yaml` (adapted from a
FLUX.2-klein full-finetune request):

* `device:` is informational — `BaseSDTrainProcess` uses
  `accelerator.device` (`xla:0` on TPU). Set it to `xla` for clarity.
* Use `optimizer: adamw` (not `adamw8bit` — the fallback handles the old
  value, but explicit is cleaner).
* Set `quantize: false`, `quantize_te: false`, `low_vram: false`,
  `layer_offloading: false`.
* Keep `dtype: bf16` (never fp16 on TPU).
* Keep `gradient_checkpointing: true` — this is your main TPU memory lever
  now that offloading is off.
* Keep resolutions **static** (e.g. `[512, 768]` only). XLA recompiles on
  every new shape; buckets cause long stalls.
* `batch_size: 1`, `gradient_accumulation: 1` to start. A 4B full finetune is
  at the edge of a single TPUv3 core (16 GiB HBM) — expect OOM without
  checkpointing + caching. LoRA (`network: {type: lora, ...}`) is strongly
  recommended on TPU; full finetune is experimental.
* `cache_latents_to_disk: true` + `cache_text_embeddings: true` + 
  `unload_text_encoder: true` all work on TPU and save HBM.
* Multi-core data-parallel is supported: set `train.tpu_num_cores: 8`
  (or pass `--tpu_cores 8` / `AITK_TPU_CORES=8`). `run.py` spawns one process
  per core via `torch_xla.distributed.xla_multiprocessing.spawn`; each core
  trains on a disjoint `file_list[ordinal::world_size]` dataset shard and
  gradients are averaged with `xm.optimizer_step`. Effective batch size is
  `batch_size × gradient_accumulation × tpu_num_cores`, so scale LR/steps
  accordingly. Latent / text-embedding / clip caches are sharded too (each
  rank caches its own files, then a rendezvous), while checkpoints, samples,
  logs and the sqlite DB stay rank-0-only.

## Limitations (honest)

* No `torch.compile` / inductor on TPU (auto-disabled).
* No fp8 / int8 quantized training on TPU (no XLA kernels in torchao/quanto).
* No CUDA-stream layer offloading equivalent — if you OOM, reduce batch /
  resolution, enable checkpointing, or switch to LoRA.
* Sampling (`sample_every`) runs on XLA and is slow; set
  `disable_sampling: true` for smoke tests.
* Model sharding (FSDP / SPMD) is not implemented — multi-core is
  data-parallel (replicated model, sharded batches). A single core's HBM
  still bounds model size; contributions welcome.
