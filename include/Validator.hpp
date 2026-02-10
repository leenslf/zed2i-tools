#pragma once
// Validation helpers for IMU and point cloud data.

#include <array>
#include <vector>

namespace zedapp {

struct ImuSample {
    std::array<float, 3> linear_accel{};
    std::array<float, 3> angular_vel{};
    std::array<float, 4> orientation{}; // x, y, z, w
};

struct PointSample {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

class Validator {
public:
    static bool validateImu(const ImuSample& sample);
    static bool validatePointCloudSamples(const std::vector<PointSample>& samples);
};

} // namespace zedapp
