#include "traversability/voxel_filter.hpp"

#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace traversability {
namespace {

struct VoxelIndex {
    std::int64_t ix;
    std::int64_t iy;
    std::int64_t iz;

    bool operator==(const VoxelIndex& other) const {
        return ix == other.ix && iy == other.iy && iz == other.iz;
    }
    bool operator<(const VoxelIndex& other) const {
        if (ix != other.ix) {
            return ix < other.ix;
        }
        if (iy != other.iy) {
            return iy < other.iy;
        }
        return iz < other.iz;
    }
};

} // namespace

VoxelFilter::VoxelFilter(const VoxelFilterConfig& config) : config_(config) {}

Eigen::MatrixXf VoxelFilter::process(const Eigen::MatrixXf& points) const {
    if (points.cols() < 3) {
        throw std::invalid_argument(
            "`points` must be 2D with at least 3 columns, got cols=" +
            std::to_string(points.cols()));
    }

    Eigen::MatrixXf xyz = points.leftCols(3);
    if (xyz.size() == 0) {
        return Eigen::MatrixXf(0, 3);
    }

    const float vx = config_.voxel_size_x;
    const float vy = config_.voxel_size_y;
    const float vz = config_.voxel_size_z;
    const int min_points = config_.min_points_per_voxel;

    if (vx <= 0.0f || vy <= 0.0f || vz <= 0.0f) {
        throw std::invalid_argument("Voxel sizes must be positive.");
    }
    if (min_points < 1) {
        throw std::invalid_argument("`min_points_per_voxel` must be >= 1.");
    }

    std::map<VoxelIndex, int> voxel_counts;

    for (Eigen::Index i = 0; i < xyz.rows(); ++i) {
        const float x = xyz(i, 0);
        const float y = xyz(i, 1);
        const float z = xyz(i, 2);

        const VoxelIndex idx{
            static_cast<std::int64_t>(std::floor(x / vx)),
            static_cast<std::int64_t>(std::floor(y / vy)),
            static_cast<std::int64_t>(std::floor(z / vz)),
        };
        ++voxel_counts[idx];
    }

    std::vector<VoxelIndex> kept_voxels;
    kept_voxels.reserve(voxel_counts.size());
    for (const auto& kv : voxel_counts) {
        if (kv.second >= min_points) {
            kept_voxels.push_back(kv.first);
        }
    }

    if (kept_voxels.empty()) {
        return Eigen::MatrixXf(0, 3);
    }

    Eigen::MatrixXf centered_xyz(static_cast<Eigen::Index>(kept_voxels.size()), 3);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(kept_voxels.size()); ++i) {
        const VoxelIndex& idx = kept_voxels[static_cast<std::size_t>(i)];
        centered_xyz(i, 0) = (static_cast<float>(idx.ix) + 0.5f) * vx;
        centered_xyz(i, 1) = (static_cast<float>(idx.iy) + 0.5f) * vy;
        centered_xyz(i, 2) = (static_cast<float>(idx.iz) + 0.5f) * vz;
    }

    return centered_xyz;
}

} // namespace traversability
