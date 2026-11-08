#include "options.h"
#include <charconv>
#include <stdexcept>
#include <vector>

namespace upscaler {
const char* usage()
{
    return R"(Usage: upscaler INPUT (--live | --anime) [options] OUTPUT.png

  --live          RealESRGAN_x4plus for photographs and live-action images
  --anime         APISR DAT-GAN for anime and illustrations
  --scale 1|2|4     Output scale (default: 4; 1/2 use GPU downsampling of 4x)
  --tile N        Tile core size, multiple of 16; 0 forces full frame
  --halo N        Context halo, multiple of 16 (live: 96, anime: 48)
  --raw FILE      Save unclipped HWC float32 output for validation
  --decoded FILE  Save decoded RGB input as PNG for validation
  --trace DIR     Save intermediate tensors; requires full-frame inference
  --help          Show this help
  --version       Show the version

Examples:
  upscaler input.jpg --live --scale 2 output.png
  upscaler input.png --anime --scale 4 output.png

Both models and all shaders are embedded. Output is always an RGB PNG.
)";
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    bool selected = false, positionalOnly = false;
    std::vector<std::string> paths;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (!positionalOnly && arg == "--")
        {
            positionalOnly = true;
            continue;
        }
        if (positionalOnly || arg.empty() || arg[0] != '-')
        {
            paths.push_back(arg);
            continue;
        }
        if (arg == "--help" || arg == "-h")
        {
            options.help = true;
            continue;
        }
        if (arg == "--version")
        {
            options.version = true;
            continue;
        }
        if (arg == "--live" || arg == "--anime")
        {
            Model model = arg == "--live" ? Model::Live : Model::Anime;
            if (selected && options.model != model) throw std::invalid_argument("Choose exactly one of --live and --anime");
            options.model = model;
            selected = true;
            continue;
        }
        const bool numeric = arg == "--scale" || arg == "--tile" || arg == "--halo";
        if (!numeric && arg != "--raw" && arg != "--decoded" && arg != "--trace") throw std::invalid_argument("Unknown option: " + arg);
        if (++i >= argc) throw std::invalid_argument("Missing value for " + arg);
        const std::string value = argv[i];
        if (numeric)
        {
            int number;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), number);
            if (result.ec != std::errc() || result.ptr != value.data() + value.size()) throw std::invalid_argument("Invalid integer for " + arg + ": " + value);
            if (arg == "--scale") options.scale = number;
            if (arg == "--tile")
            {
                if (number < 0) throw std::invalid_argument("Tile must be 0 or a positive multiple of 16");
                options.tile = number;
            }
            if (arg == "--halo")
            {
                if (number < 16) throw std::invalid_argument("Halo must be a positive multiple of 16");
                options.halo = number;
            }
        }
        else if (arg == "--raw")
            options.raw = value;
        else if (arg == "--decoded")
            options.decoded = value;
        else
            options.trace = value;
    }
    if (options.help || options.version) return options;
    if (!selected) throw std::invalid_argument("Choose --live or --anime");
    if (paths.size() != 2) throw std::invalid_argument("Provide exactly one input and one output path");
    if (options.scale != 1 && options.scale != 2 && options.scale != 4) throw std::invalid_argument("Scale must be 1, 2 or 4");
    if (options.tile > 0 && options.tile % 16) throw std::invalid_argument("Tile must be a multiple of 16");
    if (options.halo == -1) options.halo = options.model == Model::Anime ? 48 : 96;
    if (options.halo % 16) throw std::invalid_argument("Halo must be a multiple of 16");
    // Prevent overflowing crop coordinates even for malformed command lines.
    if (options.tile > 1048576 || options.halo > 1048576) throw std::invalid_argument("Tile or halo is too large");
    if (!options.trace.empty() && options.tile > 0) throw std::invalid_argument("--trace requires --tile 0");
    options.input = paths[0];
    options.output = paths[1];
    return options;
}
} // namespace upscaler
