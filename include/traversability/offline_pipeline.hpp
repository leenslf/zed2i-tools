#pragma once

#include <Eigen/Core>

#include "traversability/config.hpp"
#include "traversability/polarize.hpp"
#include "traversability/tilt_compensate.hpp"
#include "traversability/traversability.hpp"
#include "traversability/voxel_filter.hpp"

namespace traversability {

// TODO: LivePipeline will follow the same interface with a different source.
class OfflinePipeline {
public:
    explicit OfflinePipeline(const PipelineConfig& config);
    void process_frame(const Eigen::MatrixXf& points,
                       const Eigen::Vector4f& quaternion,
                       int frame_index);

private:
    PipelineConfig config_;
    TiltCompensate tilt_compensate_;
    VoxelFilter voxel_filter_;
    Polarize polarize_;
    Traversability traversability_;
};

} // namespace traversability
