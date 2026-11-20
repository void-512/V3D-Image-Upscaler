#pragma once

#include <filesystem>

namespace upscaler {
std::filesystem::path runtimeRoot();
void selectPrivateDriver();
inline constexpr const char* privateDriverTag = "(V3D Upscaler)";
}
