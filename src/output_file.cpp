#include "output_file.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <stdexcept>
#include <sys/stat.h>
#include <system_error>
#include <utility>
#include <unistd.h>

namespace upscaler {
std::filesystem::path resolveOutputPath(std::filesystem::path path)
{
    // Preserve writing through symlinks, including links to a new file.
    for (int links = 0; is_symlink(symlink_status(path)); ++links)
    {
        if (links == 40) throw std::runtime_error("Too many symlinks for output " + path.string());
        path = path.parent_path() / read_symlink(path);
    }
    return path;
}

OutputFile::OutputFile(const std::string& path) : destination(path)
{
    if (path.empty()) return;

    const auto status = std::filesystem::status(destination);
    const bool replacing = is_regular_file(status);
    if (!exists(status) || replacing)
    {
        destination = resolveOutputPath(destination);
        std::string directory = (destination.parent_path() / ".upscaler-XXXXXX").string();
        stagingFile = directory + "/output";
        if (!mkdtemp(directory.data()))
            throw std::system_error(errno, std::generic_category(), "Cannot stage output " + path);
        std::copy(directory.begin(), directory.end(), stagingFile.begin());
        stagingDirectory.swap(directory);
    }

    // Devices and pipes keep their streaming behavior and are never removed.
    file = std::fopen(stagingFile.empty() ? destination.c_str() : stagingFile.c_str(), "wb");
    const auto mode = static_cast<mode_t>(status.permissions() & std::filesystem::perms::all);
    if (!file || (replacing && fchmod(fileno(file), mode) != 0))
    {
        const int error = errno;
        if (file) std::fclose(file);
        discard();
        throw std::system_error(error, std::generic_category(), "Cannot open output " + path);
    }
    if (stagingFile.empty()) std::setvbuf(file, nullptr, _IONBF, 0);
}

OutputFile::~OutputFile()
{
    if (file) std::fclose(file);
    discard();
}

bool OutputFile::write(const void* data, size_t size)
{
    return std::fwrite(data, 1, size, file) == size;
}

void OutputFile::close()
{
    if (!file) return;
    const bool failed = std::ferror(file);
    if (std::fclose(std::exchange(file, nullptr)) != 0 || failed)
        throw std::runtime_error("Cannot finish output " + destination.string());
}

void OutputFile::publish()
{
    if (stagingFile.empty()) return;
    if (std::rename(stagingFile.c_str(), destination.c_str()) != 0)
        throw std::system_error(errno, std::generic_category(), "Cannot publish output " + destination.string());
    stagingFile.clear();
}

void OutputFile::discard() noexcept
{
    if (!stagingFile.empty()) unlink(stagingFile.c_str());
    if (!stagingDirectory.empty()) rmdir(stagingDirectory.c_str());
}
}
