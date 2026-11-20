#include "png_writer.h"
#include "runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace upscaler {
namespace {
void bigEndian(unsigned char* p, uint32_t value)
{
    for (int i = 3; i >= 0; --i) { p[i] = value & 255; value >>= 8; }
}
}
PngWriter::PngWriter(const std::string& path, int w, int h, const std::string& rawPath)
    : file(path), raw(rawPath), row(size_t(w) * 3 + 1), compressed(65536), cachedRow(size_t(w) * 3), width(w), height(h)
{
    if (!file.isOpen()) throw std::runtime_error("Cannot write PNG output");
    const unsigned char signature[] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (!file.write(signature, sizeof(signature))) throw std::runtime_error("Cannot write PNG output");
    unsigned char header[13] = {};
    bigEndian(header, w); bigEndian(header + 4, h);
    header[8] = 8; header[9] = 2;
    chunk("IHDR", header, sizeof(header));
    if (deflateInit(&stream, 3) != Z_OK) throw std::runtime_error("Cannot initialize PNG compressor");
}
PngWriter::~PngWriter()
{
    deflateEnd(&stream);
}
void PngWriter::chunk(const char* type, const unsigned char* data, size_t size)
{
    unsigned char value[4];
    bigEndian(value, size);
    if (!file.write(value, sizeof(value)) || !file.write(type, 4) || (size && !file.write(data, size)))
        throw std::runtime_error("Cannot write PNG output");
    uLong crc = crc32(0, reinterpret_cast<const unsigned char*>(type), 4);
    if (size) crc = crc32(crc, data, size);
    bigEndian(value, crc);
    if (!file.write(value, sizeof(value))) throw std::runtime_error("Cannot write PNG output");
}
void PngWriter::compress(int flush)
{
    int status;
    do
    {
        stream.next_out = compressed.data();
        stream.avail_out = compressed.size();
        status = deflate(&stream, flush);
        if (status != Z_OK && status != Z_STREAM_END) throw std::runtime_error("PNG compression failed");
        const size_t count = compressed.size() - stream.avail_out;
        if (count) chunk("IDAT", compressed.data(), count);
    } while (stream.avail_in || !stream.avail_out || (flush == Z_FINISH && status != Z_STREAM_END));
}
void PngWriter::rows(const float* pixels, int first, int count)
{
    if (first != written || first + count > height) throw std::runtime_error("Incorrect output row order");
    for (int y = 0; y < count; ++y)
    {
        checkInterrupted();
        // Stage one mapped GPU row in host memory before quantization.
        const size_t rowBytes = cachedRow.size() * sizeof(float);
        std::memcpy(cachedRow.data(), pixels, rowBytes);
        pixels += cachedRow.size();
        if (raw.isOpen() && !raw.write(cachedRow.data(), rowBytes))
            throw std::runtime_error("Cannot write raw output");
        // PNG Sub filter: subtract the preceding pixel's matching channel.
        row[0] = 1;
        unsigned char previous[3] = {};
        for (int x = 0; x < width * 3; ++x)
        {
            const float value = cachedRow[x];
            if (!std::isfinite(value)) throw std::runtime_error("Non-finite model output");
            const auto code = static_cast<unsigned char>(std::lround(std::clamp(value, 0.f, 1.f) * 255.f));
            row[x + 1] = code - previous[x % 3];
            previous[x % 3] = code;
        }
        stream.next_in = row.data();
        stream.avail_in = row.size();
        compress(Z_NO_FLUSH);
    }
    written += count;
}
void PngWriter::finish()
{
    if (finished) return;
    if (written != height) throw std::runtime_error("Incomplete PNG output");
    checkInterrupted();
    stream.next_in = nullptr;
    stream.avail_in = 0;
    compress(Z_FINISH);
    chunk("IEND", nullptr, 0);
    file.close();
    raw.close();
    checkInterrupted();
    raw.publish();
    file.publish();
    finished = true;
}
} // namespace upscaler
