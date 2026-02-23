#include "camera/Validator.hpp"

#include <cmath>

namespace zedapp {
namespace {

bool isFinite(float value) {
    return std::isfinite(value);
}

bool isFinite3(const std::array<float, 3>& v) {
    return isFinite(v[0]) && isFinite(v[1]) && isFinite(v[2]);
}

bool isFinite4(const std::array<float, 4>& v) {
    return isFinite(v[0]) && isFinite(v[1]) && isFinite(v[2]) && isFinite(v[3]);
}

} // namespace

bool Validator::validateImu(const ImuSample& sample) {
    return isFinite3(sample.linear_accel) && isFinite3(sample.angular_vel) && isFinite4(sample.orientation);
}

bool Validator::validatePointCloudSamples(const std::vector<PointSample>& samples) {
    if (samples.empty()) {
        return false;
    }

    for (const auto& sample : samples) {
        if (isFinite(sample.x) && isFinite(sample.y) && isFinite(sample.z)) {
            return true;
        }
    }

    return false;
}

} // namespace zedapp
