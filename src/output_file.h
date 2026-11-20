#pragma once

#include <cstdio>
#include <filesystem>
#include <string>

namespace upscaler {
std::filesystem::path resolveOutputPath(std::filesystem::path path);

// Stage regular files on their destination filesystem so publishing is atomic.
class OutputFile
{
public:
    explicit OutputFile(const std::string& path);
    ~OutputFile();
    OutputFile(const OutputFile&) = delete;
    OutputFile& operator=(const OutputFile&) = delete;

    bool isOpen() const { return file != nullptr; }
    bool write(const void* data, size_t size);
    void close();
    void publish();

private:
    void discard() noexcept;

    std::filesystem::path destination;
    std::string stagingDirectory, stagingFile;
    std::FILE* file = nullptr;
};
}
