#pragma once
// Recording data streams to disk.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include <sl/Camera.hpp>

#include "Config.hpp"
#include "DataRetriever.hpp"
#include "IZedCamera.hpp"

namespace zedapp {

class DataRecorder {
public:
    explicit DataRecorder(const Config& config, IZedCamera* camera = nullptr);

    void setEnabled(bool enabled);
    void toggle();
    bool isRecording() const;

    void handleSnapshot(DataSnapshot& snapshot);

    std::optional<std::filesystem::path> sessionPath() const;

private:
    bool startSession();
    void stopSession();
    bool shouldStopForLimits() const;

    std::filesystem::path buildSessionPath() const;

    const Config& config_;
    IZedCamera* camera_ = nullptr;
    std::atomic<bool> enabled_{false};
    std::atomic<bool> recording_{false};

    std::filesystem::path session_dir_;
    std::filesystem::path pointclouds_dir_;

    std::chrono::steady_clock::time_point start_time_{};
    std::size_t recorded_frame_count_ = 0;
    std::size_t progress_tick_ = 0;
};

} // namespace zedapp
