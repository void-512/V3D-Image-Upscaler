# Pi 5 Image Upscaler

A Raspberry Pi 5 Vulkan image upscaler running on GPU.

```sh
upscaler input.jpg --live --scale 2 output.png
upscaler input.png --anime --scale 4 output.png
upscaler input.jpg --live --scale 1 restored.png
```

| Mode | Model | Intended input |
|---|---|---|
| `--live` | RealESRGAN_x4plus, 23-block RRDBNet | Photographs and live-action images |
| `--anime` | APISR DAT-GAN / DAT-Small | Anime and illustrations |

## Build

Requirements:

- ARM64 Linux on Raspberry Pi 5, with the Vulkan 1.3 V3DV runtime.
- Git, CMake, Make/Ninja, and C/C++20 compilers.
- Vulkan development files and `glslc`.
- Python with CPU PyTorch

Use a Python environment with the conversion dependencies installed. For the
existing `.venv`, the build commands are:

```sh
git submodule update --init --recursive
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build --target upscaler --parallel 6
```

## Memory and tiling

Full-frame inference preserves the complete image context. When the estimated GPU
working set exceeds 2 GiB or the anime attention estimate exceeds 1 GiB, inference uses balanced tiles with
576-pixel cores and 96-pixel halos for live mode, or 448-pixel cores and 48-pixel
halos for anime mode. Tile and halo sizes must be multiples of 16. Overlapping
outputs are feathered.

Tiled inference is approximate: anime attention depends on the whole image, and
finite halos can omit convolution context. Use `--tile 0` for full-frame inference
when memory permits. Explicit Vulkan working sets are limited to 3 GiB, and each
tensor must fit the device's storage-buffer limit.

```sh
upscaler input.png --anime --tile 0 output.png
upscaler input.jpg --live --tile 576 --halo 96 output.png
```

## Source layout

```text
src/                 CLI, image I/O, Vulkan runtime and model reader
src/networks/        DAT and RRDB inference schedules
shaders/anime/       DAT kernels
shaders/live/        RRDB kernels
models/checkpoints/  original model checkpoints
tools/              model conversion, reference architectures and asset embedding
third_party/stb/     stb image I/O submodule
third_party/zlib-ng/ statically linked compression library submodule
```
