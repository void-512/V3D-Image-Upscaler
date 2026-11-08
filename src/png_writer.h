#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <zlib.h>

namespace upscaler {
// PNG and optional unclipped floats are emitted a row at a time. Large images
// never require an entire 4x float image plus an encoded-image copy in memory.
class PngWriter
{
public:
    PngWriter(const std::string& path, int width, int height, const std::string& rawPath = {});
    ~PngWriter();
    void rows(const float* pixels, int first, int count);
    void finish();
private:
    void chunk(const char* type, const unsigned char* data, size_t size);
    void compress(int flush);
    std::ofstream file, raw;
    z_stream stream{};
    std::vector<unsigned char> row, compressed;
    std::vector<float> cachedRow;
    int width, height, written = 0;
    bool initialized = false, finished = false;
};
} // namespace upscaler
