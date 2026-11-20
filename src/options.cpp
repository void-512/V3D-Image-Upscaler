#include "options.h"
#include <charconv>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace upscaler {
namespace {
int parseInteger(const std::string& arg, const std::string& value)
{
    int number;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), number);
    if (result.ec != std::errc() || result.ptr != value.data() + value.size())
        throw std::invalid_argument("Invalid integer for " + arg + ": " + value);
    return number;
}

bool benchmarkRequested(int argc, char** argv)
{
    for (int i = 1; i < argc && std::string_view(argv[i]) != "--"; ++i)
        if (std::string_view(argv[i]) == "--benchmark") return true;
    return false;
}

Options parseBenchmark(int argc, char** argv)
{
    Options options;
    options.benchmark = true;
    std::filesystem::path configFile;
    for (int i = 1; i < argc && std::string_view(argv[i]) != "--"; ++i)
    {
        const std::string arg = argv[i];
        if (arg != "--config" && arg != "--warmup" && arg != "--loop") continue;
        if (++i >= argc || std::string_view(argv[i]) == "--")
            throw std::invalid_argument("Missing value for " + arg);
        const std::string value = argv[i];
        if (arg == "--config")
        {
            if (value.empty()) throw std::invalid_argument("--config requires a file path");
            configFile = value;
        }
        else if (arg == "--warmup")
        {
            options.warmup = parseInteger(arg, value);
            if (options.warmup < 0) throw std::invalid_argument("--warmup must be nonnegative");
        }
        else
        {
            options.loops = parseInteger(arg, value);
            if (options.loops < 1) throw std::invalid_argument("--loop must be positive");
        }
    }
    options.config = loadConfig(configFile);
    return options;
}

void applyDefaults(Task& task, const Config& config)
{
    const auto& tiling = config.tiling(task.model);
    if (task.tile == -1) task.tile = tiling.tile;
    if (task.halo == -1) task.halo = tiling.halo;
}
}

const char* usage()
{
    return R"(Usage: upscaler INPUT (--live | --anime) [options] OUTPUT.png
       upscaler -i INPUT (--live | --anime) [options] -o OUTPUT.png [-i INPUT ...]
       upscaler --benchmark [--warmup M] [--loop N] [--config FILE]
       upscaler [--config FILE]

  -i INPUT        Start a command-line task
  -o OUTPUT       Set the current task's PNG output
  --live          RealESRGAN_x4plus for photographs and live-action images
  --anime         APISR DAT-GAN for anime and illustrations
  --scale 1|2|4     Output scale (default: 4; 1/2 use GPU downsampling of 4x)
  --config FILE   Load performance settings and tasks from a TOML file
  --tile N        Tile core size, multiple of 16; 0 forces full frame
  --halo N        Context halo, multiple of 16 (live: 96, anime: 48)
  --benchmark     Time inference for both models using the configured tuning
  --warmup M      Benchmark warmup passes per model (default: 1)
  --loop N        Timed benchmark passes per model (default: 3)
  --help          Show this help
  --version       Show the version

Examples:
  upscaler input.jpg --live --scale 2 output.png
  upscaler -i input.png --anime -o anime.png -i photo.jpg --live --scale 2 -o photo.png

Models are loaded from the runtime's share/upscaler/models directory.
Without --config, config.toml is searched in the working directory, then the runtime root.
Command-line tasks run before [[tasks]] entries. Each task produces an RGB PNG.
Benchmark mode uses a built-in image and reads only --warmup, --loop and --config.
)";
}

Options parseOptions(int argc, char** argv)
{
    if (benchmarkRequested(argc, argv)) return parseBenchmark(argc, argv);

    Options options;
    Task task;
    bool selected = false, positionalOnly = false, named = false, taskOption = false;
    std::filesystem::path configFile;
    std::vector<std::string> positional;
    const auto appendTask = [&] {
        if (!selected) throw std::invalid_argument("Choose --live or --anime for each command-line task");
        if (task.input.empty() || task.output.empty())
            throw std::invalid_argument("Provide an input and an output path for each command-line task");
        options.tasks.push_back(std::move(task));
        task = Task{};
        selected = false;
    };
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
            if (named) throw std::invalid_argument("Use -i and -o for all paths when specifying command-line tasks");
            positional.push_back(arg);
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
            if (selected && task.model != model) throw std::invalid_argument("Choose exactly one of --live and --anime per task");
            task.model = model;
            selected = true;
            taskOption = true;
            continue;
        }
        const bool numeric = arg == "--scale" || arg == "--tile" || arg == "--halo";
        if (!numeric && arg != "-i" && arg != "-o" && arg != "--config")
            throw std::invalid_argument("Unknown option: " + arg);
        if (++i >= argc) throw std::invalid_argument("Missing value for " + arg);
        const std::string value = argv[i];
        if (arg == "--config")
        {
            if (value.empty()) throw std::invalid_argument("--config requires a file path");
            configFile = value;
            continue;
        }
        taskOption = true;
        if (numeric)
        {
            const int number = parseInteger(arg, value);
            if (arg == "--scale")
            {
                if (!validScale(number)) throw std::invalid_argument("Scale must be 1, 2 or 4");
                task.scale = number;
            }
            else if (arg == "--tile")
            {
                if (number < 0 || number > maxTileDimension || number % 16)
                    throw std::invalid_argument("Tile must be 0 or a positive multiple of 16 up to " + std::to_string(maxTileDimension));
                task.tile = number;
            }
            else
            {
                if (number < 16 || number > maxTileDimension || number % 16)
                    throw std::invalid_argument("Halo must be a positive multiple of 16 up to " + std::to_string(maxTileDimension));
                task.halo = number;
            }
        }
        else if (arg == "-i")
        {
            if (!positional.empty()) throw std::invalid_argument("Use -i and -o for all paths when specifying command-line tasks");
            if (named) appendTask();
            named = true;
            task.input = value;
        }
        else if (arg == "-o")
        {
            if (!named || !task.output.empty() || value.empty())
                throw std::invalid_argument("Specify one -o OUTPUT after each -i INPUT");
            task.output = value;
        }
    }
    if (options.help || options.version) return options;
    if (named)
        appendTask();
    else if (taskOption || !positional.empty())
    {
        if (positional.size() != 2) throw std::invalid_argument("Provide exactly one input and one output path");
        task.input = std::move(positional[0]);
        task.output = std::move(positional[1]);
        appendTask();
    }
    options.config = loadConfig(configFile, &options.tasks);
    if (options.tasks.empty()) throw std::invalid_argument("Provide a command-line task or add [[tasks]] to the config");
    for (auto& queued : options.tasks)
        applyDefaults(queued, options.config);
    return options;
}
} // namespace upscaler
