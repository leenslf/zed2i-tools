#include "traversability/config.hpp"

#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace traversability {

PipelineConfig load_config(const std::string& path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::BadFile&) {
        throw std::runtime_error("Failed to open config file: " + path);
    }

    PipelineConfig cfg{};
    cfg.svo.coordinate_units = root["svo"]["coordinate_units"].as<std::string>("METER");
    cfg.svo.coordinate_system =
        root["svo"]["coordinate_system"].as<std::string>("RIGHT_HANDED_Z_UP_X_FWD");
    cfg.svo.depth_mode = root["svo"]["depth_mode"].as<std::string>("PERFORMANCE");
    cfg.svo.frame_skip = root["svo"]["frame_skip"].as<int>(10);

    cfg.tilt_compensate.write_output = root["tilt_compensate"]["write_output"].as<bool>(false);
    cfg.tilt_compensate.output_dir =
        root["tilt_compensate"]["output_dir"].as<std::string>("temp-offline-outs/tilt_compensate");

    cfg.voxel_filter.voxel_size_x = root["voxel_filter"]["voxel_size_x"].as<float>(0.05f);
    cfg.voxel_filter.voxel_size_y = root["voxel_filter"]["voxel_size_y"].as<float>(0.05f);
    cfg.voxel_filter.voxel_size_z = root["voxel_filter"]["voxel_size_z"].as<float>(0.05f);
    cfg.voxel_filter.min_points_per_voxel =
        root["voxel_filter"]["min_points_per_voxel"].as<int>(8);
    cfg.voxel_filter.write_output = root["voxel_filter"]["write_output"].as<bool>(false);
    cfg.voxel_filter.output_dir =
        root["voxel_filter"]["output_dir"].as<std::string>("temp-offline-outs/voxel_filter");

    cfg.polarize.z_threshold = root["polarize"]["z_threshold"].as<float>(1.5f);
    cfg.polarize.min_range = root["polarize"]["min_range"].as<float>(0.1f);
    cfg.polarize.write_output = root["polarize"]["write_output"].as<bool>(false);
    cfg.polarize.output_dir =
        root["polarize"]["output_dir"].as<std::string>("temp-offline-outs/polarize");

    cfg.traversability.danger_threshold =
        root["traversability"]["danger_threshold"].as<float>(0.3f);
    cfg.traversability.scrit_deg = root["traversability"]["scrit_deg"].as<float>(30.0f);
    cfg.traversability.rcrit_m = root["traversability"]["rcrit_m"].as<float>(0.10f);
    cfg.traversability.hcrit_m = root["traversability"]["hcrit_m"].as<float>(0.20f);
    cfg.traversability.polar_grid_size_r_m =
        root["traversability"]["polar_grid_size_r_m"].as<float>(0.10f);
    cfg.traversability.polar_grid_size_theta_deg =
        root["traversability"]["polar_grid_size_theta_deg"].as<float>(1.0f);
    cfg.traversability.write_output = root["traversability"]["write_output"].as<bool>(true);
    cfg.traversability.output_dir =
        root["traversability"]["output_dir"].as<std::string>("temp-offline-outs/traversability");

    return cfg;
}

} // namespace traversability
