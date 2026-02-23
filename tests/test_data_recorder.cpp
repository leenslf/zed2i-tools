#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "camera/DataRecorder.hpp"
#include "MockZedCamera.hpp"

namespace zedapp {
namespace {

std::filesystem::path makeTempRoot(const std::string& name) {
    const auto base = std::filesystem::temp_directory_path();
    std::ostringstream path;
    path << name << "_" << std::chrono::steady_clock::now().time_since_epoch().count();
    return base / path.str();
}

} // namespace

TEST(DataRecorderTest, CreatesFolderStructure) {
    Config config;
    config.enable_frames = true;
    config.enable_imu = true;
    config.enable_odometry = true;
    config.enable_point_cloud = true;
    config.recording_point_cloud_format = PointCloudFormat::Ply;
    config.recording_root = makeTempRoot("zed_recordings").string();

    DataRecorder recorder(config);
    recorder.setEnabled(true);

    const auto session_path = recorder.sessionPath();
    ASSERT_TRUE(session_path.has_value());
    EXPECT_TRUE(std::filesystem::exists(*session_path));
    EXPECT_TRUE(std::filesystem::exists(*session_path / "images"));
    EXPECT_TRUE(std::filesystem::exists(*session_path / "pointclouds"));
    EXPECT_TRUE(std::filesystem::exists(*session_path / "imu.csv"));
    EXPECT_TRUE(std::filesystem::exists(*session_path / "odometry.csv"));

    recorder.setEnabled(false);
    std::filesystem::remove_all(config.recording_root);
}

TEST(DataRecorderTest, WritesImuAndOdometryCsv) {
    Config config;
    config.enable_frames = false;
    config.enable_point_cloud = false;
    config.enable_imu = true;
    config.enable_odometry = true;
    config.recording_root = makeTempRoot("zed_recordings").string();

    DataRecorder recorder(config);
    recorder.setEnabled(true);

    DataSnapshot snapshot;
    snapshot.timestamp = sl::Timestamp();
    ImuData imu;
    imu.sample.linear_accel = {1.0f, 2.0f, 3.0f};
    imu.sample.angular_vel = {0.1f, 0.2f, 0.3f};
    imu.sample.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    snapshot.imu = imu;

    OdometryData odom;
    odom.tracking_state = sl::POSITIONAL_TRACKING_STATE::OK;
    snapshot.odometry = odom;

    recorder.handleSnapshot(snapshot);

    const auto session_path = recorder.sessionPath();
    ASSERT_TRUE(session_path.has_value());
    std::ifstream imu_in(*session_path / "imu.csv");
    std::ifstream odom_in(*session_path / "odometry.csv");
    ASSERT_TRUE(imu_in.is_open());
    ASSERT_TRUE(odom_in.is_open());

    std::string line;
    std::getline(imu_in, line);
    EXPECT_FALSE(line.empty());
    std::getline(imu_in, line);
    EXPECT_FALSE(line.empty());

    std::getline(odom_in, line);
    EXPECT_FALSE(line.empty());
    std::getline(odom_in, line);
    EXPECT_FALSE(line.empty());

    recorder.setEnabled(false);
    std::filesystem::remove_all(config.recording_root);
}

TEST(DataRecorderTest, WritesFrameImages) {
    Config config;
    config.enable_frames = true;
    config.enable_imu = false;
    config.enable_odometry = false;
    config.enable_point_cloud = false;
    config.recording_image_format = ImageFormat::Png;
    config.recording_root = makeTempRoot("zed_recordings").string();

    DataRecorder recorder(config);
    recorder.setEnabled(true);

    DataSnapshot snapshot;
    snapshot.timestamp = sl::Timestamp();

    sl::Mat left(1, 1, sl::MAT_TYPE::U8_C4, sl::MEM::CPU);
    sl::Mat right(1, 1, sl::MAT_TYPE::U8_C4, sl::MEM::CPU);
    sl::uchar4 pixel{255, 0, 0, 255};
    left.setValue(0, 0, pixel);
    right.setValue(0, 0, pixel);

    snapshot.frames = FrameData{left, right};
    recorder.handleSnapshot(snapshot);

    const auto session_path = recorder.sessionPath();
    ASSERT_TRUE(session_path.has_value());
    EXPECT_TRUE(std::filesystem::exists(*session_path / "images" / "left_000000.png"));
    EXPECT_TRUE(std::filesystem::exists(*session_path / "images" / "right_000000.png"));

    recorder.setEnabled(false);
    std::filesystem::remove_all(config.recording_root);
}

TEST(DataRecorderTest, WritesPointCloudPly) {
    Config config;
    config.enable_frames = false;
    config.enable_imu = false;
    config.enable_odometry = false;
    config.enable_point_cloud = true;
    config.recording_point_cloud_format = PointCloudFormat::Ply;
    config.recording_root = makeTempRoot("zed_recordings").string();

    DataRecorder recorder(config);
    recorder.setEnabled(true);

    DataSnapshot snapshot;
    snapshot.timestamp = sl::Timestamp();

    sl::Mat cloud(1, 1, sl::MAT_TYPE::F32_C4, sl::MEM::CPU);
    sl::float4 point{1.0f, 2.0f, 3.0f, 1.0f};
    cloud.setValue(0, 0, point);
    snapshot.point_cloud = PointCloudData{cloud};

    recorder.handleSnapshot(snapshot);

    const auto session_path = recorder.sessionPath();
    ASSERT_TRUE(session_path.has_value());
    const auto ply_path = *session_path / "pointclouds" / "cloud_000000.ply";
    EXPECT_TRUE(std::filesystem::exists(ply_path));

    std::ifstream ply_in(ply_path);
    std::string header;
    std::getline(ply_in, header);
    EXPECT_EQ(header, "ply");

    recorder.setEnabled(false);
    std::filesystem::remove_all(config.recording_root);
}

TEST(DataRecorderTest, SvoSkipsImuOdomAndImages) {
    Config config;
    config.enable_frames = true;
    config.enable_imu = true;
    config.enable_odometry = true;
    config.enable_point_cloud = true;
    config.recording_point_cloud_format = PointCloudFormat::Svo;
    config.recording_image_format = ImageFormat::Png;
    config.recording_root = makeTempRoot("zed_recordings").string();

    MockZedCamera camera;
    DataRecorder recorder(config, &camera);
    recorder.setEnabled(true);

    DataSnapshot snapshot;
    snapshot.timestamp = sl::Timestamp();
    ImuData imu;
    imu.sample.linear_accel = {1.0f, 2.0f, 3.0f};
    imu.sample.angular_vel = {0.1f, 0.2f, 0.3f};
    imu.sample.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    snapshot.imu = imu;
    OdometryData odom;
    odom.tracking_state = sl::POSITIONAL_TRACKING_STATE::OK;
    snapshot.odometry = odom;
    sl::Mat left(1, 1, sl::MAT_TYPE::U8_C4, sl::MEM::CPU);
    sl::Mat right(1, 1, sl::MAT_TYPE::U8_C4, sl::MEM::CPU);
    sl::uchar4 pixel{255, 0, 0, 255};
    left.setValue(0, 0, pixel);
    right.setValue(0, 0, pixel);
    snapshot.frames = FrameData{left, right};
    recorder.handleSnapshot(snapshot);

    const auto session_path = recorder.sessionPath();
    ASSERT_TRUE(session_path.has_value());
    EXPECT_FALSE(std::filesystem::exists(*session_path / "imu.csv"));
    EXPECT_FALSE(std::filesystem::exists(*session_path / "odometry.csv"));
    EXPECT_FALSE(std::filesystem::exists(*session_path / "images" / "left_000000.png"));

    recorder.setEnabled(false);
    std::filesystem::remove_all(config.recording_root);
}

TEST(DataRecorderTest, PlyModeStillWritesImuOdomAndImages) {
    Config config;
    config.enable_frames = true;
    config.enable_imu = true;
    config.enable_odometry = true;
    config.enable_point_cloud = true;
    config.recording_point_cloud_format = PointCloudFormat::Ply;
    config.recording_image_format = ImageFormat::Png;
    config.recording_root = makeTempRoot("zed_recordings").string();

    DataRecorder recorder(config);
    recorder.setEnabled(true);

    DataSnapshot snapshot;
    snapshot.timestamp = sl::Timestamp();
    ImuData imu;
    imu.sample.linear_accel = {1.0f, 2.0f, 3.0f};
    imu.sample.angular_vel = {0.1f, 0.2f, 0.3f};
    imu.sample.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    snapshot.imu = imu;
    OdometryData odom;
    odom.tracking_state = sl::POSITIONAL_TRACKING_STATE::OK;
    snapshot.odometry = odom;
    sl::Mat left(1, 1, sl::MAT_TYPE::U8_C4, sl::MEM::CPU);
    sl::Mat right(1, 1, sl::MAT_TYPE::U8_C4, sl::MEM::CPU);
    sl::uchar4 pixel{255, 0, 0, 255};
    left.setValue(0, 0, pixel);
    right.setValue(0, 0, pixel);
    snapshot.frames = FrameData{left, right};
    recorder.handleSnapshot(snapshot);

    const auto session_path = recorder.sessionPath();
    ASSERT_TRUE(session_path.has_value());
    EXPECT_TRUE(std::filesystem::exists(*session_path / "imu.csv"));
    EXPECT_TRUE(std::filesystem::exists(*session_path / "odometry.csv"));
    EXPECT_TRUE(std::filesystem::exists(*session_path / "images" / "left_000000.png"));

    recorder.setEnabled(false);
    std::filesystem::remove_all(config.recording_root);
}

TEST(DataRecorderTest, SvoUsesLosslessCompression) {
    Config config;
    config.enable_frames = false;
    config.enable_imu = false;
    config.enable_odometry = false;
    config.enable_point_cloud = true;
    config.recording_point_cloud_format = PointCloudFormat::Svo;
    config.recording_root = makeTempRoot("zed_recordings").string();

    MockZedCamera camera;
    DataRecorder recorder(config, &camera);
    recorder.setEnabled(true);

    EXPECT_EQ(camera.last_recording_params.compression_mode, sl::SVO_COMPRESSION_MODE::LOSSLESS);

    recorder.setEnabled(false);
    std::filesystem::remove_all(config.recording_root);
}

} // namespace zedapp
