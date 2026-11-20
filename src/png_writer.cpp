#include "png_writer.h"
#include "output_file.h"
#include "runtime.h"
#include "stb_image_write.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace upscaler {
namespace {
size_t rgbSize(int width, int height)
{
    // stb counts filtered bytes and doubles its compression buffer using int.
    constexpr uint64_t maxFilteredSize = (std::numeric_limits<int>::max() / 2 - 64) * uint64_t(8) / 9;
    const uint64_t stride = uint64_t(width) * 3;
    if ((stride + 1) * height > maxFilteredSize)
        throw std::runtime_error("Output image exceeds the PNG encoder size limit");
    return stride * height;
}
}

PngWriter::PngWriter(const std::string& path, int w, int h)
    : path(path), rgb(std::make_unique_for_overwrite<unsigned char[]>(rgbSize(w, h))),
      cachedRow(size_t(w) * 3), width(w), height(h)
{
}

void PngWriter::rows(const float* pixels, int first, int count)
{
    if (first != written || count < 0 || count > height - written)
        throw std::runtime_error("Incorrect output row order");
    const size_t stride = cachedRow.size();
    auto* row = rgb.get() + size_t(first) * stride;
    for (int y = 0; y < count; ++y)
    {
        checkInterrupted();
        // Stage one mapped GPU row in host memory before quantization.
        std::memcpy(cachedRow.data(), pixels, stride * sizeof(float));
        for (size_t x = 0; x < stride; ++x)
        {
            const float value = cachedRow[x];
            if (!std::isfinite(value)) throw std::runtime_error("Non-finite model output");
            row[x] = static_cast<unsigned char>(std::lround(std::clamp(value, 0.f, 1.f) * 255.f));
        }
        pixels += stride;
        row += stride;
    }
    written += count;
}

void PngWriter::finish()
{
    if (written != height) throw std::runtime_error("Incomplete PNG output");
    checkInterrupted();
    OutputFile file(path);
    struct WriteContext { OutputFile& file; bool failed = false; } context{file};
    const auto write = [](void* user, void* data, int size) {
        auto& output = *static_cast<WriteContext*>(user);
        if (!interruptionSignal())
            output.failed = !output.file.write(data, size);
    };
    stbi_write_force_png_filter = 1; // Sub filter, as used by the row encoder.
    const int encoded = stbi_write_png_to_func(write, &context, width, height, 3, rgb.get(), width * 3);
    checkInterrupted();
    if (!encoded) throw std::runtime_error("Cannot encode PNG output");
    if (context.failed) throw std::runtime_error("Cannot write PNG output");
    file.close();
    checkInterrupted();
    file.publish();
}
} // namespace upscaler
