# DocSafe: Toward Practical Print-Proof Image Steganography via Frequency Decomposition and Covariance Alignment 

<p align="center">
  <img src="assets/DocSafe_log.png" alt="DocSafe logo" width="160">
</p>

[🌐 Project Website](https://farhadsh1992.github.io/DocSafe/) &middot;
[📄 Paper (IEEE Xplore)](https://ieeexplore.ieee.org/document/11471765) &middot;
[📦 PyPI](https://pypi.org/project/DocSafe/)

Deep-learning-based watermark encoder/decoder for images and documents, with a full
training pipeline (M1/M2/M3 model variants).

![DocSafe architecture](docs/assets/img/fig1_architecture.png)
*Figure 1. The full architecture of DocSafe consists of an encoder and a decoder. A
dedicated preprocessing module transforms the original image into gradient maps and
wavelet sub-band representations, while the message is processed through the Message
Embedding Network (MEN). The fused features pass through the U-Net and convolutional
layers of the encoder to produce a residual image, which is added to the original
image to form the encoded image. Before decoding, distortions are simulated and a
preprocessing stage enhances high-frequency components to recover the message.*

## Install

```bash
pip install DocSafe
```

Or from source:

```bash
git clone https://github.com/farhadsh1992/DocSafe.git
cd DocSafe
pip install .
```

## Pretrained weights

Model weights are **not** bundled in the PyPI package (they're hundreds of MB, well past
PyPI's practical size limits). Download `pre_trained_models/` separately and point
`path_model=` at it — see the release/host you're distributing weights from.

## Quickstart: encode / decode a watermark

```python
from DocSafe import encoder, decoder

encoder_router = encoder(model="M1", path_model="pre_trained_models/", secret_size=100)
encoder_router.load_network(device="cpu")

images = encoder_router.read_image(path=["test_images/original_images.jpg"])
image_batch = encoder_router.preprocess_images(images)
encoded = encoder_router(original_images=image_batch, messages="viste", mask=None)
encoder_router.save_encoded_image("encoded.png")

decoder_router = decoder(model="M1", path_model="pre_trained_models/", secret_size=100)
decoder_router.load_network(device="cpu")

encoded_images = decoder_router.read_image(path=["encoded.png"])
encoded_batch = decoder_router.preprocess_images(encoded_images)
messages = decoder_router(encoded_images=encoded_batch, mask=None)
print(messages)
```

See `test_main.py` for a complete runnable example.

## Training

`train_main.py` shows the intended `DocSafe.Trainer` API. Two things to know before running it:

- `configs/paths_config.py` hardcodes paths from the original training machine
  (`pretrained_models/...`, dataset directories) — edit it for your own environment.
- `Trainer.train()` and `Trainer.load_network()` are still under active development
  (see the inline `NOTE:` comments in `train_main.py`).

## Known limitations

- `DocSafe.loss_functions_lib.models.stylegan2` (and everything that depends on it —
  `hyperstyle`, `psp`, `e4e`, `restyle_e4e_encoders`, `w_encoder`, the hypernetworks) JIT-compiles
  CUDA/C++ extensions on import via `torch.utils.cpp_extension.load()`. It requires an actual
  CUDA GPU + `nvcc` + `ninja` (`pip install DocSafe[stylegan2]`) and will not import on
  CPU-only machines.
- `DocSafe.networks.net_StampOne` is superseded/legacy — it isn't used by `encoder`/`decoder`/
  `Trainer` (which all use `DocSafe.networks.networks_M1/M2/M3`) and has an unresolved
  `Deformable_Conv2D` dependency in one file.
- `DocSafe.networks.DETR_NET.segmentation`'s panoptic-segmentation helpers need
  `pip install DocSafe[detr-extras]`.

## License

MIT — see `LICENSE`.
