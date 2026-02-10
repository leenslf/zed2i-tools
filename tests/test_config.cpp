#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "Config.hpp"

namespace zedapp {

TEST(ConfigTest, ParsesConfigFile) {
    // Validates parsing of config file key/value settings.
    const std::string path = "/tmp/zed_config_test.conf";
    std::ofstream out(path);
    out << "enable_frames=false\n";
    out << "enable_imu=true\n";
    out << "enable_odometry=false\n";
    out << "enable_point_cloud=true\n";
    out << "depth_mode=ULTRA\n";
    out << "serial_number=12345\n";
    out.close();

    const auto config = Config::fromFile(path);
    EXPECT_FALSE(config.enable_frames);
    EXPECT_TRUE(config.enable_imu);
    EXPECT_FALSE(config.enable_odometry);
    EXPECT_TRUE(config.enable_point_cloud);
    EXPECT_EQ(config.depth_mode, DepthMode::Ultra);
    EXPECT_TRUE(config.serial_number.has_value());
    EXPECT_EQ(*config.serial_number, 12345u);
}

TEST(ConfigTest, OverridesApply) {
    // Ensures CLI overrides update only specified fields.
    Config base;
    base.enable_frames = true;
    base.depth_mode = DepthMode::Quality;

    ConfigOverrides overrides;
    overrides.enable_frames = false;
    overrides.depth_mode = DepthMode::Performance;

    base.applyOverrides(overrides);
    EXPECT_FALSE(base.enable_frames);
    EXPECT_EQ(base.depth_mode, DepthMode::Performance);
}

} // namespace zedapp
