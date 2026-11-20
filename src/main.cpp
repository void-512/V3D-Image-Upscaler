#include "options.h"
#include "dispatch_profile.h"
#include "runtime.h"
#include "upscale.h"
#include "png_writer.h"
#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>

using namespace upscaler;

namespace {
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

void run(int argc, char** argv)
{
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;
    Options options = parseOptions(argc, argv);
    checkInterrupted();
    if (options.help)
    {
        std::cout << usage();
        return;
    }
    if (options.version)
    {
        std::cout << "upscaler " << UPSCALER_VERSION << '\n';
        return;
    }
    if (options.driverInfo) {
        Gpu gpu(options.model, options.config);
        return;
    }
    RunTimings host;
    const auto decodeStart = Clock::now();
    int width, height, channels;
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(stbi_load(options.input.c_str(), &width, &height, &channels, 3), stbi_image_free);
    checkInterrupted();
    if (!pixels) throw std::runtime_error(std::string("Cannot decode input: ") + stbi_failure_reason());
    if (width <= 0 || height <= 0 || int64_t(width) * height > INT32_MAX / 48) throw std::runtime_error("Image dimensions too large");
    if (!options.decoded.empty()) saveDecoded(options.decoded, pixels.get(), width, height);
    host.decode = Milliseconds(Clock::now() - decodeStart).count();
    if (options.tile == -1)
    {
        const bool largeAttention = options.model == Model::Anime && uint64_t((height + 15) / 16 * 16) * ((width + 15) / 16 * 16) * 384 * 4 > uint64_t(1) * 1024 * 1024 * 1024;
        const auto memoryLimit = uint64_t(options.config.tileMemoryThresholdMiB) * 1024 * 1024;
        const bool tile = estimatedMemory(options.model, height, width, options.scale, options.config) > memoryLimit || largeAttention;
        options.tile = tile ? options.config.tiling(options.model).tileSize : 0;
    }
    if (options.tile && !options.trace.empty()) throw std::runtime_error("--trace requires --tile 0");
    if (options.tile) std::cerr << "Tiling: core=" << options.tile << ", halo=" << options.halo << ". Cropped-context output can differ from full-frame inference.\n";
    const auto driverStart = Clock::now();
    Gpu gpu(options.model, options.config);
    checkInterrupted();
    host.driver = Milliseconds(Clock::now() - driverStart).count();
    gpu.tuned = !options.referenceGraph;
    DispatchProfile timings(gpu, options.profile);
    const auto modelStart = Clock::now();
    auto network = options.model == Model::Anime ? makeAnimeNetwork(gpu) : makeLiveNetwork(gpu);
    checkInterrupted();
    host.model = Milliseconds(Clock::now() - modelStart).count();
    network->trace = options.trace;
    if (!options.tile)
    {
        const auto allocateStart = Clock::now();
        network->allocate(height, width, options.scale);
        const auto uploadStart = Clock::now();
        upload(*network, options.model, pixels.get(), width, 0, 0);
        host.allocate = Milliseconds(uploadStart - allocateStart).count();
        host.upload = Milliseconds(Clock::now() - uploadStart).count();
    }
    PngWriter writer(options.output, width * options.scale, height * options.scale, options.raw);
    upscale(*network, pixels.get(), width, height, options,
            {{options.scale, [&](const float* data, int first, int rows) {
                const auto encodeStart = Clock::now();
                writer.rows(data, first, rows);
                host.encode += Milliseconds(Clock::now() - encodeStart).count();
            }}}, options.profile.empty() ? nullptr : &host);
    const auto finishStart = Clock::now();
    writer.finish();
    host.encode += Milliseconds(Clock::now() - finishStart).count();
    if (!options.profile.empty())
        std::cerr << "Profile host," << host.decode << ',' << host.driver << ',' << host.model << ','
                  << host.allocate << ',' << host.upload << ',' << host.blend << ',' << host.encode << '\n';
}
}

int main(int argc, char** argv)
{
    int status = 0;
    try
    {
        installSignalHandlers();
        run(argc, argv);
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
