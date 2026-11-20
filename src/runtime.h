#pragma once

#include <filesystem>

namespace upscaler {
std::filesystem::path runtimeRoot();
void installSignalHandlers();
int interruptionSignal() noexcept;
void checkInterrupted();
}
