#pragma once

#include <filesystem>

namespace upscaler {
std::filesystem::path runtimeRoot();
void selectPrivateDriver();
void installSignalHandlers();
int interruptionSignal() noexcept;
void checkInterrupted();
inline constexpr const char* privateDriverTag = "(V3D Upscaler)";
}
