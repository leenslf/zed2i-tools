#include "traversability/pipeline_runner.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

namespace traversability {
namespace {

double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                  const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/// Extract all finite-coordinate points from a ZED point-cloud buffer.
///
/// The ZED SDK stores invalid depth pixels as NaN/Inf.  This helper filters
/// them out and returns a dense (N, 3) float32 matrix ready for the pipeline.
///
/// @param point_cloud  ZED Mat of type MEASURE::XYZ retrieved into CPU memory.
/// @returns            (N, 3) Eigen matrix of [X, Y, Z] in metres.
Eigen::MatrixXf extract_finite_xyz(sl::Mat& point_cloud) {
    const int width  = point_cloud.getWidth();
    const int height = point_cloud.getHeight();
    if (width <= 0 || height <= 0) {
        return Eigen::MatrixXf(0, 3);
    }

    std::vector<Eigen::Vector3f> pts;
    pts.reserve(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));

    sl::float4 p;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            point_cloud.getValue(x, y, &p);
            if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
                pts.emplace_back(p.x, p.y, p.z);
            }
        }
    }

    Eigen::MatrixXf out(static_cast<Eigen::Index>(pts.size()), 3);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(pts.size()); ++i) {
        out.row(i) = pts[static_cast<std::size_t>(i)];
    }
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// PipelineRunner
// ---------------------------------------------------------------------------

PipelineRunner::PipelineRunner(const PipelineConfig& config)
    : tilt_compensate_(config.tilt_compensate),
      voxel_filter_(config.voxel_filter),
      polarize_(config.polarize),
      traversability_(config.traversability) {}

PipelineRunner::~PipelineRunner() {
    stop();
}

void PipelineRunner::start(sl::Camera& camera, int frame_skip, int max_processed_frames) {
    stop_requested_.store(false);
    done_.store(false);
    {
        std::lock_guard<std::mutex> lk(stats_mtx_);
        stats_ = PipelineStats{};
    }
    worker_ = std::thread(
        &PipelineRunner::run, this, std::ref(camera), frame_skip, max_processed_frames);
}

void PipelineRunner::stop() {
    stop_requested_.store(true);
    // Wake a blocked next() so it can observe stop_requested_ / done_.
    slot_cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

bool PipelineRunner::next(PipelineFrame& frame, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(slot_mtx_);
    slot_cv_.wait_for(lk, timeout, [this] {
        return slot_.has_value() || done_.load();
    });
    if (!slot_.has_value()) {
        return false;
    }
    frame = std::move(*slot_);
    slot_.reset();
    return true;
}

bool PipelineRunner::done() const {
    return done_.load();
}

PipelineStats PipelineRunner::stats() const {
    std::lock_guard<std::mutex> lk(stats_mtx_);
    return stats_;
}

void PipelineRunner::run(sl::Camera& camera, int frame_skip, int max_processed_frames) {
    sl::Mat  point_cloud;
    sl::Pose pose;
    int frame_index = 0;

    while (!stop_requested_.load()) {
        const sl::ERROR_CODE err = camera.grab();

        if (err == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
            std::lock_guard<std::mutex> lk(stats_mtx_);
            stats_.reached_end_of_input = true;
            break;
        }
        if (err != sl::ERROR_CODE::SUCCESS) {
            // Non-fatal grab error (e.g. transient sensor issue): skip frame.
            ++frame_index;
            {
                std::lock_guard<std::mutex> lk(stats_mtx_);
                ++stats_.grab_failures;
            }
            continue;
        }

        // Increment first so frame_index == 1 on the very first successful grab,
        // matching the Python viewer's convention.
        ++frame_index;
        {
            const int64_t timestamp_ns = static_cast<int64_t>(
                camera.getTimestamp(sl::TIME_REFERENCE::IMAGE).getNanoseconds());
            std::lock_guard<std::mutex> lk(stats_mtx_);
            ++stats_.successful_grabs;
            if (stats_.first_frame_index == 0) {
                stats_.first_frame_index = frame_index;
                stats_.first_timestamp_ns = timestamp_ns;
            }
            stats_.last_frame_index = frame_index;
            stats_.last_timestamp_ns = timestamp_ns;
        }
        if (frame_index % frame_skip != 0) {
            continue;
        }

        // Retrieve dense point cloud into CPU-accessible memory.
        // CPU memory is used because downstream Eigen code runs on the CPU.
        const auto retrieve_start = std::chrono::steady_clock::now();
        if (camera.retrieveMeasure(point_cloud, sl::MEASURE::XYZ, sl::MEM::CPU) !=
            sl::ERROR_CODE::SUCCESS) {
            continue;
        }
        const auto retrieve_end = std::chrono::steady_clock::now();
        const double retrieve_ms = elapsed_ms(retrieve_start, retrieve_end);

        camera.getPosition(pose, sl::REFERENCE_FRAME::WORLD);
        const sl::Orientation ori = pose.getOrientation();
        // ZED quaternion order: [ox, oy, oz, ow] = [qx, qy, qz, qw]
        const Eigen::Vector4f quaternion(ori.ox, ori.oy, ori.oz, ori.ow);

        // Run the full pipeline.  Each stage is stateless and thread-safe to
        // call from a single thread.
        const auto extract_start = std::chrono::steady_clock::now();
        const Eigen::MatrixXf xyz = extract_finite_xyz(point_cloud);
        const auto extract_end = std::chrono::steady_clock::now();

        const auto tilt_start = std::chrono::steady_clock::now();
        const Eigen::MatrixXf detilted = tilt_compensate_.process(xyz, quaternion);
        const auto tilt_end = std::chrono::steady_clock::now();

        const auto voxel_start = std::chrono::steady_clock::now();
        const Eigen::MatrixXf filtered = voxel_filter_.process(detilted);
        const auto voxel_end = std::chrono::steady_clock::now();

        const auto polarize_start = std::chrono::steady_clock::now();
        const Eigen::MatrixXf polar = polarize_.process(filtered);
        const auto polarize_end = std::chrono::steady_clock::now();

        const auto traversability_start = std::chrono::steady_clock::now();
        TraversabilityResult result = traversability_.process(polar);
        const auto traversability_end = std::chrono::steady_clock::now();

        const double extract_ms = elapsed_ms(extract_start, extract_end);
        const double tilt_ms = elapsed_ms(tilt_start, tilt_end);
        const double voxel_ms = elapsed_ms(voxel_start, voxel_end);
        const double polarize_ms = elapsed_ms(polarize_start, polarize_end);
        const double traversability_ms =
            elapsed_ms(traversability_start, traversability_end);

        // Post to the drop slot.  We always overwrite the previous value so the
        // display thread never reads stale data: if it hasn't consumed the last
        // frame yet, we simply replace it with the fresher one.
        {
            std::lock_guard<std::mutex> lk(slot_mtx_);
            slot_ = PipelineFrame{std::move(result), frame_index};
        }
        {
            std::lock_guard<std::mutex> lk(stats_mtx_);
            ++stats_.processed_frames;
            stats_.total_retrieve_ms += retrieve_ms;
            stats_.total_extract_ms += extract_ms;
            stats_.total_tilt_ms += tilt_ms;
            stats_.total_voxel_ms += voxel_ms;
            stats_.total_polarize_ms += polarize_ms;
            stats_.total_traversability_ms += traversability_ms;
            stats_.last_retrieve_ms = retrieve_ms;
            stats_.last_extract_ms = extract_ms;
            stats_.last_tilt_ms = tilt_ms;
            stats_.last_voxel_ms = voxel_ms;
            stats_.last_polarize_ms = polarize_ms;
            stats_.last_traversability_ms = traversability_ms;
            if (max_processed_frames > 0 &&
                stats_.processed_frames >= max_processed_frames) {
                stop_requested_.store(true);
            }
        }
        slot_cv_.notify_one();
    }

    // Signal consumers that no more results will arrive.
    done_.store(true);
    slot_cv_.notify_all();
}

} // namespace traversability
