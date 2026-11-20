#pragma once
#include <memory>
#include <string>
#include <vector>

namespace upscaler {
class PngWriter
{
public:
    PngWriter(const std::string& path, int width, int height);
    void rows(const float* pixels, int first, int count);
    void finish();
private:
    std::string path;
    std::unique_ptr<unsigned char[]> rgb;
    std::vector<float> cachedRow;
    int width, height, written = 0;
};
} // namespace upscaler
