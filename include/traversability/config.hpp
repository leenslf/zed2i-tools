#pragma once

#include <string>

namespace traversability {

struct SvoConfig {
    std::string coordinate_units;
    std::string coordinate_system;
    std::string depth_mode;
    int frame_skip;
};

struct TiltCompensateConfig {
    bool write_output;
    std::string output_dir;
};

struct VoxelFilterConfig {
    float voxel_size_x;
    float voxel_size_y;
    float voxel_size_z;
    int min_points_per_voxel;
    bool write_output;
    std::string output_dir;
};

struct PolarizeConfig {
    float z_threshold;
    float min_range;
    bool write_output;
    std::string output_dir;
};

struct TraversabilityConfig {
    float danger_threshold;
    float scrit_deg;
    float rcrit_m;
    float hcrit_m;
    float polar_grid_size_r_m;
    float polar_grid_size_theta_deg;
    bool write_output;
    std::string output_dir;
};

struct PipelineConfig {
    SvoConfig svo;
    TiltCompensateConfig tilt_compensate;
    VoxelFilterConfig voxel_filter;
    PolarizeConfig polarize;
    TraversabilityConfig traversability;
};

// Loads pipeline_config.yaml from the given path.
// Applies hardcoded defaults first, then overrides with values from the file.
// Throws std::runtime_error with the missing field name if a required key
// is absent and has no default.
PipelineConfig load_config(const std::string& path);

} // namespace traversability
