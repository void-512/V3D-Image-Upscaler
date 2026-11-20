#include "options.h"
#include "dispatch_profile.h"
#include "upscale.h"
#include "png_writer.h"
#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>

using namespace upscaler;

int main(int argc, char** argv)
{
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;
    try
    {
        Options options = parseOptions(argc, argv);
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
        if (options.driverInfo) {
            Gpu gpu(options.model, options.config);
            return 0;
        }
        RunTimings host;
        const auto decodeStart = Clock::now();
        int width, height, channels;
        std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(stbi_load(options.input.c_str(), &width, &height, &channels, 3), stbi_image_free);
        if (!pixels) throw std::runtime_error(std::string("Cannot decode input: ") + stbi_failure_reason());
        if (width <= 0 || height <= 0 || int64_t(width) * height > INT32_MAX / 48) throw std::runtime_error("Image dimensions too large");
        if (!options.decoded.empty() && !stbi_write_png(options.decoded.c_str(), width, height, 3, pixels.get(), width * 3)) throw std::runtime_error("Cannot save decoded input");
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
        host.driver = Milliseconds(Clock::now() - driverStart).count();
        gpu.tuned = !options.referenceGraph;
        DispatchProfile timings(gpu, options.profile);
        const auto modelStart = Clock::now();
        auto network = options.model == Model::Anime ? makeAnimeNetwork(gpu) : makeLiveNetwork(gpu);
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
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        std::cerr << "Error: " << error.what() << "\nRun upscaler --help for usage.\n";
        return 2;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
