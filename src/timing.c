/**
 * @file timing.c
 * @brief Implementation of timing utilities for token generation throughput
 */

#define _POSIX_C_SOURCE 199309L
#include "timing.h"
#include <string.h>
#include <time.h>

void timing_init(ThroughputTracker *tracker) {
    memset(tracker, 0, sizeof(ThroughputTracker));
    clock_gettime(CLOCK_MONOTONIC, &tracker->start_time);
}

void timing_record_token(
    ThroughputTracker *tracker,
    int generated_tokens,
    float *overall_tps,
    float *window_tps) {

    struct timespec current_time;
    clock_gettime(CLOCK_MONOTONIC, &current_time);

    // Calculate overall throughput
    float elapsed = (current_time.tv_sec - tracker->start_time.tv_sec) +
                    (current_time.tv_nsec - tracker->start_time.tv_nsec) / 1e9;

    if (elapsed > 0 && generated_tokens > 0) {
        *overall_tps = generated_tokens / elapsed;
    } else {
        *overall_tps = 0.0f;
    }

    // Calculate windowed throughput
    *window_tps = 0.0f;

    if (generated_tokens > 0) {
        // Store current timestamp in circular buffer
        tracker->token_times[tracker->token_time_idx] = current_time;
        tracker->token_time_idx = (tracker->token_time_idx + 1) % TIMING_WINDOW_SIZE;
        if (tracker->token_time_count < TIMING_WINDOW_SIZE) {
            tracker->token_time_count++;
        }

        // Calculate throughput for the window (need at least 2 timestamps)
        if (tracker->token_time_count >= 2) {
            // Get the oldest timestamp in the window
            int oldest_idx = (tracker->token_time_idx - tracker->token_time_count + TIMING_WINDOW_SIZE) % TIMING_WINDOW_SIZE;
            struct timespec oldest_time = tracker->token_times[oldest_idx];

            float window_elapsed = (current_time.tv_sec - oldest_time.tv_sec) +
                                  (current_time.tv_nsec - oldest_time.tv_nsec) / 1e9;

            if (window_elapsed > 0) {
                // Number of tokens in the window (counting intervals, not timestamps)
                *window_tps = (tracker->token_time_count - 1) / window_elapsed;
            }
        }
    }
}
