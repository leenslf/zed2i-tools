#include "DataRecorder.hpp"

#include <cmath>
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <type_traits>
#include <vector>

#include "Logger.hpp"
#include "Validator.hpp"

namespace zedapp {
namespace {

template <typename T>
bool isWriteOk(const T& status) {
    if constexpr (std::is_same_v<T, sl::ERROR_CODE>) {
        return status == sl::ERROR_CODE::SUCCESS;
    }
    return static_cast<bool>(status);
}

} // namespace

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

    ++grab_count_;
    const int stride = std::max(1, config_.recording_frame_stride);
    if (stride > 1 && (grab_count_ % static_cast<std::size_t>(stride)) != 0u) {
        return;
    }

    const auto frame_index = recorded_frame_count_;
    bool wrote_any = false;

    if (config_.enable_frames && snapshot.frames.has_value()) {
        wrote_any |= writeFrames(*snapshot.frames, frame_index);
    }

    if (config_.enable_imu && snapshot.imu.has_value()) {
        wrote_any |= writeImu(*snapshot.imu, snapshot.timestamp);
    }

    if (config_.enable_odometry && snapshot.odometry.has_value()) {
        wrote_any |= writeOdometry(*snapshot.odometry, snapshot.timestamp);
    }

    if (config_.enable_point_cloud && snapshot.point_cloud.has_value()) {
        wrote_any |= writePointCloud(*snapshot.point_cloud, frame_index);
    }

    if (wrote_any) {
        ++recorded_frame_count_;
        if (++progress_tick_ % 50 == 0) {
            std::ostringstream out;
            out << "Recording frame " << recorded_frame_count_;
            if (config_.recording_frame_limit > 0) {
                out << "/" << config_.recording_frame_limit;
            }
            Logger::log(LogLevel::Info, out.str());
        }
    }
}

bool DataRecorder::startSession() {
    session_dir_ = buildSessionPath();
    images_dir_ = session_dir_ / "images";
    pointclouds_dir_ = session_dir_ / "pointclouds";

    std::error_code ec;
    std::filesystem::create_directories(session_dir_, ec);
    if (ec) {
        Logger::log(LogLevel::Error, "Failed to create recording directory.");
        recording_.store(false, std::memory_order_relaxed);
        enabled_.store(false, std::memory_order_relaxed);
        return false;
    }

    if (config_.enable_frames) {
        std::filesystem::create_directories(images_dir_, ec);
    }
    if (config_.enable_point_cloud) {
        std::filesystem::create_directories(pointclouds_dir_, ec);
    }

    if (config_.enable_imu) {
        imu_csv_.open(session_dir_ / "imu.csv", std::ios::out);
        if (imu_csv_.is_open()) {
            imu_csv_ << "timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,orientation_x,orientation_y,"
                        "orientation_z,orientation_w\n";
            imu_csv_.flush();
        }
    }

    if (config_.enable_odometry) {
        odom_csv_.open(session_dir_ / "odometry.csv", std::ios::out);
        if (odom_csv_.is_open()) {
            odom_csv_ << "timestamp,pos_x,pos_y,pos_z,orientation_x,orientation_y,orientation_z,orientation_w,vel_x,vel_y,"
                         "vel_z\n";
            odom_csv_.flush();
        }
    }

    grab_count_ = 0;
    recorded_frame_count_ = 0;
    progress_tick_ = 0;
    warned_bad_odometry_ = false;
    warned_bad_pointcloud_ = false;
    start_time_ = std::chrono::steady_clock::now();
    recording_.store(true, std::memory_order_relaxed);

    if (config_.enable_point_cloud && config_.recording_point_cloud_format == PointCloudFormat::Svo) {
        if (!camera_) {
            Logger::log(LogLevel::Warn, "SVO recording requested but no camera handle available.");
        } else {
            const auto svo_path = (pointclouds_dir_ / "pointcloud.svo").string();
            sl::RecordingParameters params;
            params.video_filename = svo_path.c_str();
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
    if (config_.enable_point_cloud && config_.recording_point_cloud_format == PointCloudFormat::Svo && camera_) {
        camera_->disableRecording();
    }
    if (imu_csv_.is_open()) {
        imu_csv_.close();
    }
    if (odom_csv_.is_open()) {
        odom_csv_.close();
    }
    recording_.store(false, std::memory_order_relaxed);
    enabled_.store(false, std::memory_order_relaxed);
    Logger::log(LogLevel::Info, "Recording stopped.");
}

bool DataRecorder::shouldStopForLimits() const {
    if (config_.recording_frame_limit > 0 &&
        recorded_frame_count_ >= static_cast<std::size_t>(config_.recording_frame_limit)) {
        return true;
    }
    if (config_.recording_duration_sec > 0) {
        const auto elapsed = std::chrono::steady_clock::now() - start_time_;
        const auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
        if (elapsed_sec >= config_.recording_duration_sec) {
            return true;
        }
    }
    return false;
}

bool DataRecorder::writeFrames(FrameData& frames, std::size_t frame_index) {
    if (frames.left.getWidth() == 0 || frames.left.getHeight() == 0) {
        return false;
    }
    if (frames.right.getWidth() == 0 || frames.right.getHeight() == 0) {
        return false;
    }

    const auto extension = (config_.recording_image_format == ImageFormat::Jpg) ? "jpg" : "png";
    std::ostringstream left_name;
    left_name << "left_" << std::setfill('0') << std::setw(6) << frame_index << "." << extension;
    std::ostringstream right_name;
    right_name << "right_" << std::setfill('0') << std::setw(6) << frame_index << "." << extension;

    const auto left_path = (images_dir_ / left_name.str()).string();
    const auto right_path = (images_dir_ / right_name.str()).string();

    const auto left_ok = isWriteOk(frames.left.write(left_path.c_str()));
    const auto right_ok = isWriteOk(frames.right.write(right_path.c_str()));

    return left_ok && right_ok;
}

bool DataRecorder::writeImu(const ImuData& imu, sl::Timestamp timestamp) {
    if (!imu_csv_.is_open()) {
        return false;
    }
    if (!Validator::validateImu(imu.sample)) {
        Logger::log(LogLevel::Warn, "Skipping invalid IMU sample.");
        return false;
    }

    imu_csv_ << formatTimestamp(timestamp) << ','
             << imu.sample.linear_accel[0] << ',' << imu.sample.linear_accel[1] << ',' << imu.sample.linear_accel[2]
             << ',' << imu.sample.angular_vel[0] << ',' << imu.sample.angular_vel[1] << ','
             << imu.sample.angular_vel[2] << ',' << imu.sample.orientation[0] << ',' << imu.sample.orientation[1]
             << ',' << imu.sample.orientation[2] << ',' << imu.sample.orientation[3] << '\n';
    imu_csv_.flush();
    return true;
}

bool DataRecorder::writeOdometry(const OdometryData& odometry, sl::Timestamp timestamp) {
    if (!odom_csv_.is_open()) {
        return false;
    }
    if (odometry.tracking_state != sl::POSITIONAL_TRACKING_STATE::OK) {
        if (!warned_bad_odometry_) {
            Logger::log(LogLevel::Warn, "Skipping odometry sample with bad tracking state.");
            warned_bad_odometry_ = true;
        }
        return false;
    }

    sl::Pose pose = odometry.pose;
    const auto translation = pose.getTranslation();
    const auto orientation = pose.getOrientation();
    const auto velocity = pose.twist;

    odom_csv_ << formatTimestamp(timestamp) << ','
              << translation.x << ',' << translation.y << ',' << translation.z << ','
              << orientation.ox << ',' << orientation.oy << ',' << orientation.oz << ',' << orientation.ow << ','
              << velocity[0] << ',' << velocity[1] << ',' << velocity[2] << '\n';
    odom_csv_.flush();
    return true;
}

bool DataRecorder::writePointCloud(PointCloudData& cloud, std::size_t frame_index) {
    if (config_.recording_point_cloud_format == PointCloudFormat::Svo) {
        return true;
    }
    if (cloud.cloud.getWidth() == 0 || cloud.cloud.getHeight() == 0) {
        return false;
    }

    if (!hasValidPointSample(cloud.cloud)) {
        if (!warned_bad_pointcloud_) {
            Logger::log(LogLevel::Warn, "Skipping invalid point cloud sample.");
            warned_bad_pointcloud_ = true;
        }
        return false;
    }

    std::ostringstream filename;
    filename << "cloud_" << std::setfill('0') << std::setw(6) << frame_index << ".ply";
    const auto path = pointclouds_dir_ / filename.str();
    return writePointCloudPly(cloud.cloud, path);
}

bool DataRecorder::writePointCloudPly(const sl::Mat& cloud, const std::filesystem::path& path) const {
    std::ofstream out(path, std::ios::out);
    if (!out.is_open()) {
        return false;
    }

    const int width = cloud.getWidth();
    const int height = cloud.getHeight();
    const std::size_t vertex_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

    out << "ply\nformat ascii 1.0\n";
    out << "element vertex " << vertex_count << "\n";
    out << "property float x\nproperty float y\nproperty float z\n";
    out << "end_header\n";

    sl::float4 point;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            cloud.getValue(x, y, &point);
            out << point.x << ' ' << point.y << ' ' << point.z << '\n';
        }
    }
    return true;
}

bool DataRecorder::hasValidPointSample(const sl::Mat& cloud) const {
    const int width = cloud.getWidth();
    const int height = cloud.getHeight();
    if (width <= 0 || height <= 0) {
        return false;
    }

    const int stride_x = std::max(1, width / 8);
    const int stride_y = std::max(1, height / 8);

    sl::float4 point;
    std::vector<PointSample> samples;
    samples.reserve(8);
    for (int y = 0; y < height; y += stride_y) {
        for (int x = 0; x < width; x += stride_x) {
            cloud.getValue(x, y, &point);
            samples.push_back(PointSample{point.x, point.y, point.z, point.w});
            if (samples.size() >= 8) {
                return Validator::validatePointCloudSamples(samples);
            }
        }
    }

    return Validator::validatePointCloudSamples(samples);
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

std::string DataRecorder::formatTimestamp(sl::Timestamp timestamp) const {
    const auto nanos = timestamp.getNanoseconds();
    return std::to_string(nanos);
}

} // namespace zedapp
