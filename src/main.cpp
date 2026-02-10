#include <chrono>
#include <csignal>
#include <atomic>
#include <cctype>
#include <thread>

#include "CameraManager.hpp"
#include "Config.hpp"
#include "DataRecorder.hpp"
#include "DataRetriever.hpp"
#include "Logger.hpp"

namespace zedapp {
namespace {

volatile std::sig_atomic_t g_running = 1;

void handleSignal(int) {
    g_running = 0;
}

int parseIterations(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--iterations=", 0) == 0) {
            return std::stoi(arg.substr(std::string("--iterations=").size()));
        }
    }
    return 0;
}

int parseSleepMs(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--sleep-ms=", 0) == 0) {
            return std::stoi(arg.substr(std::string("--sleep-ms=").size()));
        }
    }
    return 5;
}

} // namespace
} // namespace zedapp

int main(int argc, char** argv) {
    using namespace zedapp;

    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    ConfigOverrides overrides = ConfigOverrides::fromArgs(argc, argv);
    Config config;

    if (overrides.config_path.has_value()) {
        config = Config::fromFile(*overrides.config_path);
    }
    config.applyOverrides(overrides);

    CameraManager manager;
    auto camera = manager.openCamera(config);
    if (!camera) {
        return 1;
    }

    DataRetriever retriever(*camera, config);
    DataRecorder recorder(config, camera.get());

    const int iterations = parseIterations(argc, argv);
    const int sleep_ms = parseSleepMs(argc, argv);
    int count = 0;

    std::atomic<bool> recording_enabled{config.enable_recording};
    if (recording_enabled.load()) {
        recorder.setEnabled(true);
    }

    std::thread input_thread;
    if (config.recording_keyboard_toggle) {
        input_thread = std::thread([&]() {
            std::string line;
            while (g_running && std::getline(std::cin, line)) {
                if (line.empty()) {
                    continue;
                }
                const char command =
                    static_cast<char>(std::tolower(static_cast<unsigned char>(line.front())));
                if (command == 'r') {
                    recording_enabled.store(!recording_enabled.load());
                    recorder.setEnabled(recording_enabled.load());
                } else if (command == 'q') {
                    g_running = 0;
                }
            }
        });
        input_thread.detach();
    }

    while (g_running) {
        DataSnapshot snapshot;
        const auto status = retriever.retrieve(snapshot);
        if (status != sl::ERROR_CODE::SUCCESS) {
            Logger::log(LogLevel::Warn, "Grab failed, retrying.");
        }

        if (status == sl::ERROR_CODE::SUCCESS && recording_enabled.load()) {
            recorder.handleSnapshot(snapshot);
        }

        if (iterations > 0 && ++count >= iterations) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }

    recorder.setEnabled(false);

    if (config.enable_odometry) {
        camera->disablePositionalTracking();
    }
    camera->close();

    return 0;
}
