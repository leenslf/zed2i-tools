#include "traversability/polarize.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace traversability {

Polarize::Polarize(const PolarizeConfig& config) : config_(config) {}

Eigen::MatrixXf Polarize::process(const Eigen::MatrixXf& points) const {
    if (points.cols() < 3) {
        throw std::invalid_argument(
            "`points` must be 2D with at least 3 columns, got cols=" +
            std::to_string(points.cols()));
    }

    const Eigen::MatrixXf xyz = points.leftCols(3);
    if (xyz.size() == 0) {
        return Eigen::MatrixXf(0, 3);
    }

    const float z_threshold = config_.z_threshold;
    const float min_range = config_.min_range;

    if (z_threshold < 0.0f) {
        throw std::invalid_argument("`z_threshold` must be >= 0.");
    }
    if (min_range < 0.0f) {
        throw std::invalid_argument("`min_range` must be >= 0.");
    }

    std::vector<Eigen::Vector3f> z_filtered;
    z_filtered.reserve(static_cast<std::size_t>(xyz.rows()));
    for (Eigen::Index i = 0; i < xyz.rows(); ++i) {
        const float x = xyz(i, 0);
        const float y = xyz(i, 1);
        const float z = xyz(i, 2);
        if (std::abs(z) < z_threshold) {
            z_filtered.emplace_back(x, y, z);
        }
    }

    if (z_filtered.empty()) {
        return Eigen::MatrixXf(0, 3);
    }

    std::vector<Eigen::Vector3f> polar_filtered;
    polar_filtered.reserve(z_filtered.size());
    for (const Eigen::Vector3f& p : z_filtered) {
        const float x = p.x();
        const float y = p.y();
        const float z = p.z();
        const float r = std::sqrt(x * x + y * y);
        const float theta = std::atan2(y, x);

        if (r > min_range) {
            polar_filtered.emplace_back(r, theta, z);
        }
    }

    if (polar_filtered.empty()) {
        return Eigen::MatrixXf(0, 3);
    }

    Eigen::MatrixXf out(static_cast<Eigen::Index>(polar_filtered.size()), 3);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(polar_filtered.size()); ++i) {
        out.row(i) = polar_filtered[static_cast<std::size_t>(i)];
    }
    return out;
}

} // namespace traversability
