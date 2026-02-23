#pragma once
// Recording data streams to disk.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
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

    bool writeFrames(FrameData& frames, std::size_t frame_index);
    bool writeImu(const ImuData& imu, sl::Timestamp timestamp);
    bool writeOdometry(const OdometryData& odometry, sl::Timestamp timestamp);
    bool writePointCloud(PointCloudData& cloud, std::size_t frame_index);

    bool writePointCloudPly(const sl::Mat& cloud, const std::filesystem::path& path) const;
    bool hasValidPointSample(const sl::Mat& cloud) const;

    std::filesystem::path buildSessionPath() const;
    std::string formatTimestamp(sl::Timestamp timestamp) const;

    const Config& config_;
    IZedCamera* camera_ = nullptr;
    std::atomic<bool> enabled_{false};
    std::atomic<bool> recording_{false};

    std::filesystem::path session_dir_;
    std::filesystem::path images_dir_;
    std::filesystem::path pointclouds_dir_;

    std::ofstream imu_csv_;
    std::ofstream odom_csv_;

    std::chrono::steady_clock::time_point start_time_{};
    std::size_t grab_count_ = 0;
    std::size_t recorded_frame_count_ = 0;
    std::size_t progress_tick_ = 0;
    bool warned_bad_odometry_ = false;
    bool warned_bad_pointcloud_ = false;
};

} // namespace zedapp
