#include "camera/DataRecorder.hpp"

#include <ctime>
#include <iomanip>
#include <sstream>

#include "camera/Logger.hpp"

namespace zedapp {

DataRecorder::DataRecorder(const Config& config, IZedCamera* camera) : config_(config), camera_(camera) {}

void DataRecorder::setEnabled(bool enabled) {
    enabled_.store(enabled, std::memory_order_relaxed);
    if (enabled) {
        if (!recording_.load(std::memory_order_relaxed)) {
            startSession();
        }
    } else if (recording_.load(std::memory_order_relaxed)) {
        stopSession();
    }
}

void DataRecorder::toggle() {
    setEnabled(!enabled_.load(std::memory_order_relaxed));
}

bool DataRecorder::isRecording() const {
    return recording_.load(std::memory_order_relaxed);
}

std::optional<std::filesystem::path> DataRecorder::sessionPath() const {
    if (!recording_.load(std::memory_order_relaxed)) {
        return std::nullopt;
    }
    return session_dir_;
}

void DataRecorder::handleSnapshot(DataSnapshot& snapshot) {
    if (!recording_.load(std::memory_order_relaxed)) {
        return;
    }
    if (shouldStopForLimits()) {
        stopSession();
        return;
    }
    ++recorded_frame_count_;
    if (++progress_tick_ % 50 == 0) {
        std::ostringstream out;
        out << "Recording frame " << recorded_frame_count_;
        Logger::log(LogLevel::Info, out.str());
    }
}

bool DataRecorder::startSession() {
    session_dir_ = buildSessionPath();
    pointclouds_dir_ = session_dir_ / "pointclouds";

    std::error_code ec;
    std::filesystem::create_directories(session_dir_, ec);
    if (ec) {
        Logger::log(LogLevel::Error, "Failed to create recording directory.");
        recording_.store(false, std::memory_order_relaxed);
        enabled_.store(false, std::memory_order_relaxed);
        return false;
    }

    std::filesystem::create_directories(pointclouds_dir_, ec);

    recorded_frame_count_ = 0;
    progress_tick_ = 0;
    start_time_ = std::chrono::steady_clock::now();
    recording_.store(true, std::memory_order_relaxed);

    if (config_.enable_point_cloud) {
        if (!camera_) {
            Logger::log(LogLevel::Warn, "SVO recording requested but no camera handle available.");
        } else {
            const auto svo_path = (pointclouds_dir_ / "RECORDING.svo").string();
            sl::RecordingParameters params;
            params.video_filename = svo_path.c_str();
            params.compression_mode = sl::SVO_COMPRESSION_MODE::LOSSLESS;
            const auto status = camera_->enableRecording(params);
            if (status != sl::ERROR_CODE::SUCCESS) {
                Logger::log(LogLevel::Warn, "Failed to start SVO recording.");
            }
        }
    }

    Logger::log(LogLevel::Info, "Recording started.");
    return true;
}

void DataRecorder::stopSession() {
    if (config_.enable_point_cloud && camera_) {
        camera_->disableRecording();
    }
    recording_.store(false, std::memory_order_relaxed);
    enabled_.store(false, std::memory_order_relaxed);
    Logger::log(LogLevel::Info, "Recording stopped.");
}

bool DataRecorder::shouldStopForLimits() const {
    if (config_.recording_duration_sec > 0) {
        const auto elapsed = std::chrono::steady_clock::now() - start_time_;
        const auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
        if (elapsed_sec >= config_.recording_duration_sec) {
            return true;
        }
    }
    return false;
}

std::filesystem::path DataRecorder::buildSessionPath() const {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now_time);
#else
    localtime_r(&now_time, &tm);
#endif

    std::ostringstream name;
    name << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return std::filesystem::path(config_.recording_root) / name.str();
}

} // namespace zedapp
