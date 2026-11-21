#include "options.h"
#include "output_file.h"
#include "runtime.h"
#include "upscale.h"
#include "png_writer.h"
#include "stb_image.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>

using namespace upscaler;

namespace {
bool confirmPaths(const std::vector<Task>& tasks)
{
    namespace fs = std::filesystem;
    for (const auto& task : tasks)
    {
        checkInterrupted();
        const auto status = fs::status(task.input);
        if (!fs::exists(status) || fs::is_directory(status))
            throw std::runtime_error("Input file is missing or is a directory: " + task.input);
    }

    std::set<fs::path> destinations;
    std::vector<std::string> existing;
    for (const auto& task : tasks)
    {
        checkInterrupted();
        const auto& name = task.output;
        const auto status = fs::status(name);
        if (fs::is_directory(status))
            throw std::runtime_error("Output path is a directory: " + name);
        // Pipes and devices are streamed to without replacing the file itself.
        if (fs::exists(status) && !fs::is_regular_file(status)) continue;
        const auto target = fs::absolute(resolveOutputPath(name));
        if (!fs::is_directory(target.parent_path()))
            throw std::runtime_error("Output directory does not exist: " + target.parent_path().string());
        if (!destinations.insert(fs::weakly_canonical(target)).second)
            throw std::runtime_error("Multiple outputs target the same path: " + name);
        if (fs::exists(status) || fs::is_symlink(fs::symlink_status(name)))
            existing.push_back(name);
    }
    if (existing.empty()) return true;

    for (size_t i = 0; i < existing.size(); ++i)
        std::cerr << (i == 0 ? "" : ", ") << existing[i];
    std::cerr << " already exist" << (existing.size() == 1 ? "s" : "")
              << ", replace and continue? [y/N]\n" << std::flush;
    std::string answer;
    std::getline(std::cin, answer);
    checkInterrupted();
    return answer == "y" || answer == "Y";
}

void runBenchmark(const Options& options)
{
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;
    constexpr int scale = 4;
    const int width = options.benchmarkWidth, height = options.benchmarkHeight;
    const size_t stride = size_t(width) * 3;
    std::vector<unsigned char> pixels(stride * height);
    // Use a fixed seed so timing comparisons can reuse the same workload.
    std::mt19937 random;
    std::uniform_int_distribution<int> channel(0, 255);
    for (int y = 0; y < height; ++y)
    {
        checkInterrupted();
        std::generate_n(pixels.data() + y * stride, stride,
                        [&] { return static_cast<unsigned char>(channel(random)); });
    }

    const auto& config = options.config;
    std::cout << "Benchmark " << width << 'x' << height << " RGB at " << scale << "x, warmup=" << options.warmup
              << ", loops=" << options.loops << ", workgroup=" << config.linearWorkgroupSize << std::endl;
    for (const auto model : {Model::Anime, Model::Live})
    {
        checkInterrupted();
        Task task;
        task.model = model;
        task.scale = scale;
        task.tile = config.tiling(model).tile;
        task.halo = config.tiling(model).halo;
        task.tile = resolveTile(task, config, width, height);
        Gpu gpu(model, config);
        auto network = model == Model::Anime ? makeAnimeNetwork(gpu) : makeLiveNetwork(gpu);
        if (!task.tile)
        {
            network->allocate(height, width, scale);
            upload(*network, model, pixels.data(), width, 0, 0);
        }
        const auto infer = [&] {
            upscale(*network, pixels.data(), width, height, task);
        };
        for (int i = 0; i < options.warmup; ++i)
            infer();

        std::vector<double> samples;
        for (int i = 0; i < options.loops; ++i)
        {
            const auto start = Clock::now();
            infer();
            samples.push_back(Milliseconds(Clock::now() - start).count());
        }
        std::sort(samples.begin(), samples.end());
        const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        const double median = (samples[(samples.size() - 1) / 2] + samples[samples.size() / 2]) / 2;
        std::cout << modeName(model) << " (tile=" << task.tile << ", halo=" << task.halo << ")"
                  << std::fixed << std::setprecision(2) << " min=" << samples.front()
                  << " ms, median=" << median << " ms, mean=" << mean
                  << " ms, max=" << samples.back() << " ms" << std::endl;
    }
}

void runTask(Task task, const Config& config)
{
    checkInterrupted();
    int width, height;
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
        stbi_load(task.input.c_str(), &width, &height, nullptr, 3), stbi_image_free);
    checkInterrupted();
    if (!pixels) throw std::runtime_error(std::string("Cannot decode input: ") + stbi_failure_reason());
    if (!validImageDimensions(width, height))
        throw std::runtime_error("Image dimensions too large");
    task.tile = resolveTile(task, config, width, height);
    if (task.tile)
        std::cerr << "Tiling: core=" << task.tile << ", halo=" << task.halo
                  << ". Cropped-context output can differ from full-frame inference.\n";

    PngWriter writer(task.output, width * task.scale, height * task.scale);
    {
        Gpu gpu(task.model, config);
        auto network = task.model == Model::Anime ? makeAnimeNetwork(gpu) : makeLiveNetwork(gpu);
        if (!task.tile)
        {
            network->allocate(height, width, task.scale);
            upload(*network, task.model, pixels.get(), width, 0, 0);
        }
        upscale(*network, pixels.get(), width, height, task,
                [&](const float* data, int first, int rows) { writer.rows(data, first, rows); });
    }
    pixels.reset();
    writer.finish();
}

int run(int argc, char** argv)
{
    const Options options = parseOptions(argc, argv);
    checkInterrupted();
    if (options.benchmark)
    {
        runBenchmark(options);
        return 0;
    }
    if (options.help)
    {
        std::cout << usage();
        return 0;
    }
    if (options.version)
    {
        std::cout << "upscaler " << UPSCALER_VERSION << '\n';
        return 0;
    }
    if (!confirmPaths(options.tasks))
    {
        std::cerr << "Aborted.\n";
        return 1;
    }
    for (size_t i = 0; i < options.tasks.size(); ++i)
    {
        const auto& task = options.tasks[i];
        if (options.tasks.size() > 1)
            std::cerr << "Task " << i + 1 << '/' << options.tasks.size() << ": " << task.input << " -> " << task.output << '\n';
        runTask(task, options.config);
    }
    return 0;
}
}

int main(int argc, char** argv)
{
    int status = 0;
    try
    {
        installSignalHandlers();
        status = run(argc, argv);
    }
    catch (const std::invalid_argument& error)
    {
        if (!interruptionSignal())
            std::cerr << "Error: " << error.what() << "\nRun upscaler --help for usage.\n";
        status = 2;
    }
    catch (const std::exception& error)
    {
        if (!interruptionSignal())
            std::cerr << "Error: " << error.what() << '\n';
        status = 1;
    }
    if (const int signal = interruptionSignal())
    {
        std::cerr << "Interrupted by signal " << signal << '\n';
        return 128 + signal;
    }
    return status;
}
