#include "options.h"
#include "dispatch_profile.h"
#include "runtime.h"
#include "upscale.h"
#include "png_writer.h"
#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
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
    const auto inspect = [&](const std::string& name, bool directory = false) {
        if (name.empty()) return;
        checkInterrupted();
        const auto status = fs::status(name);
        if (fs::exists(status) && fs::is_directory(status) != directory)
            throw std::runtime_error("Invalid output path: " + name);
        // Pipes and devices are streamed to without replacing the file itself.
        if (!directory && fs::exists(status) && !fs::is_regular_file(status)) return;
        const auto target = fs::absolute(resolveOutputPath(name));
        if (!directory && !fs::is_directory(target.parent_path()))
            throw std::runtime_error("Output directory does not exist: " + target.parent_path().string());
        if (!destinations.insert(fs::weakly_canonical(target)).second)
            throw std::runtime_error("Multiple outputs target the same path: " + name);
        if (directory ? fs::exists(status) && !fs::is_empty(name)
                      : fs::exists(status) || fs::is_symlink(fs::symlink_status(name)))
            existing.push_back(name);
    };
    for (const auto& task : tasks)
    {
        inspect(task.output);
        inspect(task.raw);
        inspect(task.decoded);
        inspect(task.profile);
        inspect(task.trace, true);
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

void saveDecoded(const std::string& path, const unsigned char* pixels, int width, int height)
{
    OutputFile output(path);
    const auto write = [](void* context, void* data, int size) {
        static_cast<OutputFile*>(context)->write(data, size);
    };
    if (!stbi_write_png_to_func(write, &output, width, height, 3, pixels, width * 3))
        throw std::runtime_error("Cannot save decoded input");
    output.close();
    checkInterrupted();
    output.publish();
}

void runTask(Task task, const Config& config)
{
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;
    checkInterrupted();
    RunTimings host;
    const auto decodeStart = Clock::now();
    int width, height, channels;
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(stbi_load(task.input.c_str(), &width, &height, &channels, 3), stbi_image_free);
    checkInterrupted();
    if (!pixels) throw std::runtime_error(std::string("Cannot decode input: ") + stbi_failure_reason());
    if (width <= 0 || height <= 0 || int64_t(width) * height > INT32_MAX / 48) throw std::runtime_error("Image dimensions too large");
    if (!task.decoded.empty()) saveDecoded(task.decoded, pixels.get(), width, height);
    host.decode = Milliseconds(Clock::now() - decodeStart).count();
    if (task.tile == -1)
    {
        const bool largeAttention = task.model == Model::Anime && uint64_t((height + 15) / 16 * 16) * ((width + 15) / 16 * 16) * 384 * 4 > uint64_t(1) * 1024 * 1024 * 1024;
        const auto memoryLimit = uint64_t(config.tileMemoryThresholdMiB) * 1024 * 1024;
        const bool tile = estimatedMemory(task.model, height, width, task.scale, config) > memoryLimit || largeAttention;
        task.tile = tile ? config.tiling(task.model).tileSize : 0;
    }
    if (task.tile && !task.trace.empty()) throw std::runtime_error("--trace requires --tile 0");
    if (task.tile) std::cerr << "Tiling: core=" << task.tile << ", halo=" << task.halo << ". Cropped-context output can differ from full-frame inference.\n";
    const auto driverStart = Clock::now();
    Gpu gpu(task.model, config);
    checkInterrupted();
    host.driver = Milliseconds(Clock::now() - driverStart).count();
    gpu.tuned = !task.referenceGraph;
    DispatchProfile timings(gpu, task.profile);
    const auto modelStart = Clock::now();
    auto network = task.model == Model::Anime ? makeAnimeNetwork(gpu) : makeLiveNetwork(gpu);
    checkInterrupted();
    host.model = Milliseconds(Clock::now() - modelStart).count();
    network->trace = task.trace;
    if (!task.tile)
    {
        const auto allocateStart = Clock::now();
        network->allocate(height, width, task.scale);
        const auto uploadStart = Clock::now();
        upload(*network, task.model, pixels.get(), width, 0, 0);
        host.allocate = Milliseconds(uploadStart - allocateStart).count();
        host.upload = Milliseconds(Clock::now() - uploadStart).count();
    }
    PngWriter writer(task.output, width * task.scale, height * task.scale, task.raw);
    upscale(*network, pixels.get(), width, height, task,
            {{task.scale, [&](const float* data, int first, int rows) {
                const auto encodeStart = Clock::now();
                writer.rows(data, first, rows);
                host.encode += Milliseconds(Clock::now() - encodeStart).count();
            }}}, task.profile.empty() ? nullptr : &host);
    const auto finishStart = Clock::now();
    writer.finish();
    host.encode += Milliseconds(Clock::now() - finishStart).count();
    if (!task.profile.empty())
        std::cerr << "Profile host," << host.decode << ',' << host.driver << ',' << host.model << ','
                  << host.allocate << ',' << host.upload << ',' << host.blend << ',' << host.encode << '\n';
}

int run(int argc, char** argv)
{
    const Options options = parseOptions(argc, argv);
    checkInterrupted();
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
    if (options.driverInfo)
    {
        Gpu gpu(options.driverModel, options.config);
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
