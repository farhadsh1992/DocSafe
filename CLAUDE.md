# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DocSafe (DocFace) is a PyTorch/TensorFlow deep-learning watermark encoder/decoder for
images and documents, with three trained model variants (M1/M2/M3) and a (partially
unfinished) training pipeline. Packaged for PyPI as `DocSafe`.

## Environment setup

This project uses `uv`, not plain pip/conda, even though `environment2.yml` (legacy
conda spec) is still present for reference.

```bash
uv venv DocSafe1 --python 3.9
source DocSafe1/bin/activate        # or activate.fish for fish shell
uv pip install -r requirement.txt
```

`requirement.txt` has the full install/GPU-vs-CPU-torch instructions written as comments
at the bottom of the file — read those before troubleshooting install issues.

There is no test suite, linter, or Makefile in this repo. `test_main.py` is a manual,
runnable usage example (not a pytest suite) for the encode/decode path.

## Building the PyPI package

```bash
uv build            # produces dist/*.whl and dist/*.tar.gz
uv publish          # or: twine upload dist/*  (requires a PyPI account/token)
```

Packaging is defined in `pyproject.toml` ([tool.setuptools.packages.find] `include`
list) and `MANIFEST.in`. **Pretrained weights are deliberately excluded** from the
distribution (`pre_trained_models/` is ~600MB, far past practical PyPI size limits) —
users download them separately and point `path_model=` at that directory.

## Architecture

### The package is five top-level, co-dependent packages, not just `DocSafe`

`DocSafe` imports from sibling top-level packages using **absolute** imports
(`from FarhadCV.Tools import ...`, `from Network_Libs.ArtCoder.utils import ...`), not
relative ones. All of these must be present alongside `DocSafe/` for anything to import:

- **`FarhadCV/`** — general utilities: `tcolors`/`bcolors` terminal coloring used
  throughout for log output, `Tools.py` file helpers, `ErrorBinaryCode.py` (BCH +
  Reed-Solomon encoding of the watermark payload string into bits), `Tools_send_notifaction.py`.
- **`Network_Libs/`** — trimmed vendored helper nets: `ArtCoder` (style/texture loss
  helpers) and `Vision_Transformer` (patch embedding for Riemann losses). This is a
  small subset of a much larger original `Network_Libs`; only what's actually imported
  was copied in.
- **`Tools_GAN/`** — `linear_interpolation.py`, `ranger.py` (Ranger optimizer).
- **`Tools_nvidia_torch/`** — `torch_utils.py`: `Configure_GPU`, `check_free_space`,
  `CHECK_PYTHON_SETTING` (debug-only; lazily imports `pkg_resources`).
- **`configs/`** — `paths_config.py` defines `model_paths` (a dict — do not turn this
  back into a function; every caller does `model_paths["key"]`) and hardcodes paths
  from the original training machine (`/media/ssd2_data/...`, `pretrained_models/...`).
  **Must be edited per environment before training.**

### Inference path: `encoder` / `decoder` (`encoder_router.py`, `decoder_router.py`)

The documented, working entry point (see `test_main.py`). Both classes take
`model="M1"|"M2"|"M3"` which selects: the pretrained checkpoint filenames/step count
(e.g. M1 = `StampOneDetr2_SIRIN5_CompleteFFT_X5010`, step 364000) and which
`DocSafe/networks/networks_M{1,2,3}/` folder's `AttentionVNet_encoder`/`AttentionVNet_decoder`
to load. Message payloads go through BCH error-correction coding
(`FarhadCV/ErrorBinaryCode.py`, `BCH_BITS`/`BCH_POLYNOMIAL`/`number_zeros` params) before
being embedded/extracted as an image residual.

### Training path: `Trainer` (`CustomFit.py`) — under active development

`Trainer` is the class-based successor to the older free-function `train_steps.py::main_test`
(same method names: `train_multi_gpus`, `train_with_one_gpu`, `upload_live_monitor`,
`upload_augmentation`). Known unfinished parts:
- `Trainer.train(rank, world_size)` still references undefined names ported over from
  `main_test` (`args` instead of `self.args`, `ddp_setup`, `max_epochs`) — will raise
  `NameError` until finished.
- `Trainer` has no `load_network()` dispatcher (unlike `encoder`/`decoder`); call
  `Load_M1_networks()`/`Load_M2_networks()`/`Load_M3_networks()` directly.
- `train_main.py` documents the intended calling convention with inline `NOTE:` comments
  at each gap — check there before assuming a method signature.

`train_steps.py` is a standalone-script-style module (parses argv at import time via
`getArgsInputs()`) meant to be run as `python -m DocSafe.train_steps`, not imported.

### `DocSafe/networks/` has duplicated, versioned model code — only some of it is live

`networks_M1/`, `networks_M2/`, `networks_M3/` are near-duplicate per-model-variant network
definitions (encoder/decoder, spectral/stega discriminators, wavelet transforms, Snake
activation). `AffineTransform/` is shared. **`net_StampOne/` is dead/superseded code** —
it isn't imported by `encoder`, `decoder`, or `Trainer` (they all use `networks_M{1,2,3}`),
and one of its files has an unresolved `Deformable_Conv2D` import. Don't wire it in without
checking whether it's actually meant to be revived.

### `loss_functions_lib/` mixes original loss code with vendored external repos

Original: `face_id_loss.py`, `moco_loss.py`, `Riemann_Loss_*.py`, `vgg_loss.py`,
`yuv_loss.py`, `ms_ssim.py`, `QRSimulateLoss2.py`. Vendored (from public research repos,
each using **absolute** `DocSafe.loss_functions_lib.models.X` imports since they assume
`models` sits on `sys.path` as top-level): `models/` (hyperstyle, psp, e4e, stylegan2,
mtcnn), `query_selected_attention_tools/`.

**`models/stylegan2/op/{fused_act,upfirdn2d}.py` JIT-compile CUDA/C++ extensions at
import time** via `torch.utils.cpp_extension.load()`. Importing these (or anything
depending on them: `hyperstyle`, `psp`, `e4e`, `restyle_e4e_encoders`, `w_encoder`,
`hypernetworks`) requires an actual CUDA GPU + `nvcc` + `ninja` — they will not import
on a CPU-only machine, and that's inherent to the vendored code, not a bug to fix here.

`query_selected_attention_tools/qs_model.py` has broken imports and is unreachable dead
code (only referenced in doc-comment URLs elsewhere) — leave it alone.

### Supporting subpackages

- **`args/`** — `paramters.py` (exposed as `train_setting`), `paramters_noise.py`
  (`noise_setting`), `paramters_detr.py`. All argparse-based.
- **`augmentors/`** — `Augmentor_v01.py`, `warper_Augmentor_v01.py` (needs `torchgeometry`
  for perspective warping).
- **`datasets/`** — only `Loaddataset_v7.py` (`Dataset_Router`) is live; v1–v6 existed
  historically and were intentionally not carried forward.
- **`monitors/`** — `custom_callback.py`'s `Live_Monitoring` (a `keras.callbacks.Callback`)
  writes TensorBoard summaries during training.

### Optional extras

`pyproject.toml` declares `DocSafe[stylegan2]` (installs `ninja`, still needs a real CUDA
GPU) and `DocSafe[detr-extras]` (installs `panopticapi` for
`networks/DETR_NET/segmentation.py`'s panoptic helpers).
