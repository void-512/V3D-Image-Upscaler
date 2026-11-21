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

The Mesa submodule pins version 26.2.7 under `third_party/mesa`. The build applies the V3DV patch to a temporary copy to pack two FP16 values into one FP32 datapath. Set `UPSCALER_MESA_SOURCE` to use another clean Mesa source tree.

The build looks for local checkpoints in `models/checkpoints/` and downloads missing ones. Install the [Python conversion dependencies](tools/requirements.txt) to build from checkpoints. Set `UPSCALER_MODELS_DIR` to use checkpoints or prepacked weights from another directory.

```sh
git submodule update --init --recursive --depth 1
cmake -S . -B build -G Ninja
cmake --build build
```

The default build compiles both targets. Use `cmake --build build --target upscaler` to build only the executable or `cmake --build build --target driver` to build only the Mesa driver.

`build/product/` contains `upscaler` and `libvulkan_broadcom.so`. Keep them together.

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
