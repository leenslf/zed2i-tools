#include <chrono>
#include <filesystem>
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
    config.enable_point_cloud = true;
    config.recording_root = makeTempRoot("zed_recordings").string();

    MockZedCamera camera;
    DataRecorder recorder(config, &camera);
    recorder.setEnabled(true);

    const auto session_path = recorder.sessionPath();
    ASSERT_TRUE(session_path.has_value());
    EXPECT_TRUE(std::filesystem::exists(*session_path));
    EXPECT_TRUE(std::filesystem::exists(*session_path / "pointclouds"));

    recorder.setEnabled(false);
    std::filesystem::remove_all(config.recording_root);
}

TEST(DataRecorderTest, SvoUsesLosslessCompression) {
    Config config;
    config.enable_point_cloud = true;
    config.recording_root = makeTempRoot("zed_recordings").string();

    MockZedCamera camera;
    DataRecorder recorder(config, &camera);
    recorder.setEnabled(true);

    EXPECT_EQ(camera.last_recording_params.compression_mode, sl::SVO_COMPRESSION_MODE::LOSSLESS);

    recorder.setEnabled(false);
    std::filesystem::remove_all(config.recording_root);
}

} // namespace zedapp
