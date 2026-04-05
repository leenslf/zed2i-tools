#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>

#include <sl/Camera.hpp>

#include "traversability/config.hpp"
#include "traversability/polarize.hpp"
#include "traversability/tilt_compensate.hpp"
#include "traversability/traversability.hpp"
#include "traversability/voxel_filter.hpp"

namespace traversability {

/// A pipeline result paired with the source camera frame number.
/// frame_index reflects the raw grab() counter, so it increments by frame_skip
/// between successive pipeline outputs.
struct PipelineFrame {
    TraversabilityResult result;
    int frame_index{0};
};

/// Aggregate counters and timestamps collected while the pipeline is running.
struct PipelineStats {
    int64_t successful_grabs{0};
    int64_t grab_failures{0};
    int64_t processed_frames{0};
    int first_frame_index{0};
    int last_frame_index{0};
    int64_t first_timestamp_ns{0};
    int64_t last_timestamp_ns{0};
    bool reached_end_of_input{false};
    double total_retrieve_ms{0.0};
    double total_extract_ms{0.0};
    double total_tilt_ms{0.0};
    double total_voxel_ms{0.0};
    double total_polarize_ms{0.0};
    double total_traversability_ms{0.0};
    double last_retrieve_ms{0.0};
    double last_extract_ms{0.0};
    double last_tilt_ms{0.0};
    double last_voxel_ms{0.0};
    double last_polarize_ms{0.0};
    double last_traversability_ms{0.0};
};

/// Runs the full traversability pipeline in a background thread and exposes
/// the latest result to a consumer thread via a single-slot drop queue.
///
/// "Drop" semantics: if the consumer has not read the previous result before
/// the next one is produced, the previous result is silently overwritten.
/// This guarantees the display always shows the freshest data at the cost of
/// potentially skipping frames under high CPU load.
///
/// Typical usage:
/// @code
///   traversability::PipelineRunner runner(config);
///   runner.start(zed, frame_skip);
///   while (!runner.done()) {
///       traversability::PipelineFrame frame;
///       if (runner.next(frame, std::chrono::milliseconds(50)))
///           display(frame);
///       handle_input();
///   }
///   runner.stop();
///   zed.close();
/// @endcode
///
/// Thread-safety: start(), stop(), next(), and done() may be called from the
/// consumer thread.  The producer thread only calls internal members.
class PipelineRunner {
public:
    /// Constructs the runner and initialises all pipeline stages from config.
    /// The write_output flags in each stage config are ignored — this runner
    /// never writes intermediate results to disk.
    explicit PipelineRunner(const PipelineConfig& config);

    /// Destructor signals stop and waits for the worker thread to finish.
    ~PipelineRunner();

    // Non-copyable and non-movable: owns a thread and a mutex.
    PipelineRunner(const PipelineRunner&)            = delete;
    PipelineRunner& operator=(const PipelineRunner&) = delete;

    /// Launch the background producer thread.
    ///
    /// @param camera     An open, tracking-enabled sl::Camera.  The caller
    ///                   retains ownership; the camera must stay valid until
    ///                   stop() returns.
    /// @param frame_skip Process every Nth grabbed frame (1 = every frame).
    ///                   Skipped frames are grabbed but immediately discarded.
    void start(sl::Camera& camera, int frame_skip, int max_processed_frames = -1);

    /// Signal the producer to stop and block until it has exited.
    /// Safe to call more than once and from the destructor.
    void stop();

    /// Block up to `timeout` waiting for a fresh pipeline result.
    ///
    /// @param[out] frame  Populated with the latest result on success.
    /// @param      timeout  Maximum wait duration.
    /// @returns true when a result was available (frame is valid).
    ///          false on timeout or when the source is exhausted.
    bool next(PipelineFrame& frame, std::chrono::milliseconds timeout);

    /// Returns true once the producer thread has exited — either because the
    /// SVO file ended or stop() was called.  The consumer should keep draining
    /// next() until this returns true and next() returns false.
    bool done() const;

    /// Return a snapshot of the counters collected by the worker thread.
    PipelineStats stats() const;

private:
    /// Entry point for the background thread.
    void run(sl::Camera& camera, int frame_skip, int max_processed_frames);

    // ---- Pipeline stages ----
    // Constructed once in the constructor, reused for every frame.
    // Each stage is stateless across frames, so no synchronisation is needed.
    TiltCompensate tilt_compensate_;
    VoxelFilter    voxel_filter_;
    Polarize       polarize_;
    Traversability traversability_;

    // ---- Single-slot drop queue ----
    // The producer writes to slot_ under slot_mtx_ and notifies slot_cv_.
    // The consumer waits on slot_cv_, moves the slot contents out, and resets it.
    // Overwriting slot_ without consuming the old value is intentional.
    std::mutex                    slot_mtx_;
    std::condition_variable       slot_cv_;
    std::optional<PipelineFrame>  slot_;

    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> done_{false};

    mutable std::mutex stats_mtx_;
    PipelineStats stats_;

    std::thread worker_;
};

} // namespace traversability
