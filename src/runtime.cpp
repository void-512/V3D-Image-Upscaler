#include "runtime.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace upscaler {
std::filesystem::path runtimeRoot()
{
    return std::filesystem::read_symlink("/proc/self/exe").parent_path().parent_path();
}

void selectPrivateDriver()
{
    const auto root = runtimeRoot();
    const auto manifest = root / "share/vulkan/icd.d/broadcom_custom.json";
    const auto library = root / "lib/libvulkan_broadcom.so";
    for (const auto& path : {manifest, library})
        if (!std::filesystem::is_regular_file(path))
            throw std::runtime_error("Private V3DV runtime file missing: " + path.string());
    if (manifest.string().find(':') != std::string::npos)
        throw std::runtime_error("The Vulkan loader cannot address a private ICD path containing ':'");
    if (setenv("VK_DRIVER_FILES", manifest.c_str(), 1) || setenv("VK_ICD_FILENAMES", manifest.c_str(), 1))
        throw std::runtime_error("Could not configure private Vulkan driver discovery");
    // These inherited filters would otherwise be able to exclude the sole ICD.
    unsetenv("VK_LOADER_DRIVERS_SELECT");
    unsetenv("VK_LOADER_DRIVERS_DISABLE");
    std::cerr << "Private Vulkan ICD " << manifest << '\n';
}
}
