# Pi Image Upscaler

A Raspberry Pi Vulkan image upscaler running on GPU.

```sh
upscaler INPUT.jpg --live --scale 2 OUTPUT.png
upscaler INPUT.png --anime OUTPUT.png
```

| Mode | Model | Intended input |
|---|---|---|
| `--live` | RealESRGAN_x4plus, 23-block RRDBNet | Photographs and live-action images |
| `--anime` | APISR DAT-GAN / DAT-Small | Anime and illustrations |

## Build

- C/C++20 toolchain, Git, CMake 3.20+, Ninja, Meson, pkg-config, Vulkan headers, and `glslc`.
- Python with Mako and packaging.
- Development libraries for libdrm, libudev, Expat, and Zstandard.

The project uses a customized mesa that packs 2 fp16 data into a single fp32 datapath to achieve 2x fp16 performance. So it will need its own vulkan driver.

The build looks for local checkpoints in `models/checkpoints/` and downloads missing ones. Install the [Python conversion dependencies](tools/requirements.txt) to build from checkpoints. Set `UPSCALER_MODELS_DIR` to use checkpoints or prepacked weights from another directory.

```sh
git submodule update --init --recursive
cmake -S . -B build -G Ninja
cmake --build build
```

`build/product/` contains `upscaler` and `libvulkan_broadcom.so`. Keep them together. The executable embeds the models and shaders.

## Configuration

Use [config.toml](config.toml) to set inference options or queue images. Pass `--config FILE` to select a file. By default, the program looks in the working directory, then beside the executable.

Repeat `-i INPUT --live -o OUTPUT` or `-i INPUT --anime -o OUTPUT` to queue tasks. Command-line tasks run before entries from the config. Paths in the config are relative to that file.

## Benchmark

```sh
upscaler --benchmark --width 128 --height 96 --warmup 1 --loop 3
```

Both models benchmark 4× inference on the same in-memory RGB image, generated from a fixed random seed. Supply both dimensions on the command line. The config still controls inference tuning such as tiling and workgroup size.

## Memory and tiling

Automatic tiling limits GPU memory use. Tiling can change the output where a tile has less image context. Use `--tile 0` for full-frame inference when memory permits. `--tile` and `--halo` override the config.
