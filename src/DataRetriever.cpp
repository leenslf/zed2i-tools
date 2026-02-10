#include "DataRetriever.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "Logger.hpp"

namespace zedapp {

DataRetriever::DataRetriever(IZedCamera& camera, const Config& config)
    : camera_(camera), config_(config) {}

sl::ERROR_CODE DataRetriever::retrieve(DataSnapshot& out_snapshot) {
    const auto grab_status = camera_.grab(runtime_params_);
    if (grab_status != sl::ERROR_CODE::SUCCESS) {
        return grab_status;
    }

    out_snapshot = DataSnapshot{};
    out_snapshot.timestamp = camera_.getTimestamp(sl::TIME_REFERENCE::IMAGE);

    if (config_.enable_frames) {
        const auto left_status = camera_.retrieveImage(left_, sl::VIEW::LEFT);
        const auto right_status = camera_.retrieveImage(right_, sl::VIEW::RIGHT);
        if (left_status == sl::ERROR_CODE::SUCCESS && right_status == sl::ERROR_CODE::SUCCESS) {
            out_snapshot.frames = FrameData{left_, right_};
        }
    }

    if (config_.enable_imu) {
        const auto sensors_status = camera_.getSensorsData(sensors_data_, sl::TIME_REFERENCE::IMAGE);
        if (sensors_status == sl::ERROR_CODE::SUCCESS) {
            ImuData imu_data;
            imu_data.sample = toImuSample(sensors_data_.imu);
            imu_data.timestamp = sensors_data_.imu.timestamp;
            out_snapshot.imu = imu_data;

            if (!Validator::validateImu(imu_data.sample)) {
                Logger::log(LogLevel::Warn, "IMU validation failed.");
            }
        }
    }

    if (config_.enable_odometry) {
        const auto state = camera_.getPosition(pose_, sl::REFERENCE_FRAME::WORLD);
        out_snapshot.odometry = OdometryData{pose_, state};
    }

    if (config_.enable_point_cloud) {
        const auto cloud_status = camera_.retrieveMeasure(point_cloud_, sl::MEASURE::XYZRGBA);
        if (cloud_status == sl::ERROR_CODE::SUCCESS) {
            out_snapshot.point_cloud = PointCloudData{point_cloud_};

            const auto samples = samplePointCloud(point_cloud_);
            if (!Validator::validatePointCloudSamples(samples)) {
                Logger::log(LogLevel::Warn, "Point cloud validation failed.");
            }
        }
    }

    return sl::ERROR_CODE::SUCCESS;
}

ImuSample DataRetriever::toImuSample(const sl::SensorsData::IMUData& imu) const {
    ImuSample sample;

    sample.linear_accel = {imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z};
    sample.angular_vel = {imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z};

    const auto orientation = imu.pose.getOrientation();
    sample.orientation = {orientation.ox, orientation.oy, orientation.oz, orientation.ow};

    return sample;
}

std::vector<PointSample> DataRetriever::samplePointCloud(const sl::Mat& cloud) const {
    std::vector<PointSample> samples;
    const int width = cloud.getWidth();
    const int height = cloud.getHeight();
    if (width <= 0 || height <= 0) {
        return samples;
    }

    const int stride_x = std::max(1, width / 8);
    const int stride_y = std::max(1, height / 8);

    sl::float4 point;
    for (int y = 0; y < height; y += stride_y) {
        for (int x = 0; x < width; x += stride_x) {
            cloud.getValue(x, y, &point);
            const PointSample sample{point.x, point.y, point.z, point.w};
            if (std::isfinite(sample.x) && std::isfinite(sample.y) && std::isfinite(sample.z)) {
                samples.push_back(sample);
                if (samples.size() >= 8) {
                    return samples;
                }
            }
        }
    }

    return samples;
}

} // namespace zedapp
