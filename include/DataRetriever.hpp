#pragma once
// Data acquisition from the ZED camera.

#include <optional>

#include <sl/Camera.hpp>

#include "Config.hpp"
#include "IZedCamera.hpp"
#include "Validator.hpp"

namespace zedapp {

struct FrameData {
    sl::Mat left;
    sl::Mat right;
};

struct ImuData {
    ImuSample sample;
    sl::Timestamp timestamp;
};

struct OdometryData {
    sl::Pose pose;
    sl::POSITIONAL_TRACKING_STATE tracking_state = sl::POSITIONAL_TRACKING_STATE::OFF;
};

struct PointCloudData {
    sl::Mat cloud;
};

struct DataSnapshot {
    std::optional<FrameData> frames;
    std::optional<ImuData> imu;
    std::optional<OdometryData> odometry;
    std::optional<PointCloudData> point_cloud;
};

class DataRetriever {
public:
    DataRetriever(IZedCamera& camera, const Config& config);

    sl::ERROR_CODE retrieve(DataSnapshot& out_snapshot);

private:
    ImuSample toImuSample(const sl::SensorsData::IMUData& imu) const;
    std::vector<PointSample> samplePointCloud(const sl::Mat& cloud) const;

    IZedCamera& camera_;
    const Config& config_;

    sl::RuntimeParameters runtime_params_{};
    sl::SensorsData sensors_data_{};
    sl::Pose pose_{};

    sl::Mat left_{};
    sl::Mat right_{};
    sl::Mat point_cloud_{};
};

} // namespace zedapp
