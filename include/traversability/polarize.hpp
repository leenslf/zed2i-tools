#pragma once

#include <Eigen/Core>

#include "traversability/config.hpp"

namespace traversability {

// Points is (N, C) where C >= 3; only the first three columns (XYZ) are used.
// Returns (M, 3) float32 array of [r, theta, z] polar points.
// Throws std::invalid_argument on malformed input or invalid config values.
class Polarize {
public:
    explicit Polarize(const PolarizeConfig& config);
    Eigen::MatrixXf process(const Eigen::MatrixXf& points) const;

private:
    PolarizeConfig config_;
};

} // namespace traversability
