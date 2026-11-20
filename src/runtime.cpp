#include "runtime.h"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <stdexcept>
#include <system_error>

namespace upscaler {
namespace {
static_assert(std::atomic<int>::is_always_lock_free);
std::atomic<int> receivedSignal{0};

void recordSignal(int signal)
{
    int expected = 0;
    receivedSignal.compare_exchange_strong(expected, signal, std::memory_order_relaxed);
}
}

void installSignalHandlers()
{
    struct sigaction action{};
    action.sa_handler = recordSignal;
    sigemptyset(&action.sa_mask);
    for (const int signal : {SIGTERM, SIGINT, SIGHUP})
        if (sigaction(signal, &action, nullptr) != 0)
            throw std::system_error(errno, std::generic_category(), "Cannot install signal handler");
}

int interruptionSignal() noexcept
{
    return receivedSignal.load(std::memory_order_relaxed);
}

void checkInterrupted()
{
    if (interruptionSignal())
        throw std::runtime_error("Interrupted");
}

std::filesystem::path runtimeRoot()
{
    return std::filesystem::read_symlink("/proc/self/exe").parent_path();
}
}
