/**
 * @file timing.h
 * @brief Timing utilities for measuring token generation throughput
 *
 * This file provides utilities for tracking overall and windowed throughput
 * during token generation.
 */

#pragma once
#include <time.h>

/** @brief Default window size for recent throughput calculation */
#define TIMING_WINDOW_SIZE 4

/**
 * @struct ThroughputTracker
 * @brief Structure for tracking token generation throughput
 *
 * This structure maintains timing information to calculate both overall
 * throughput and recent throughput over a sliding window.
 */
typedef struct {
    /** @brief Start time of token generation */
    struct timespec start_time;
    
    /** @brief Circular buffer for recent token timestamps */
    struct timespec token_times[TIMING_WINDOW_SIZE];
    
    /** @brief Current write position in circular buffer */
    int token_time_idx;
    
    /** @brief Number of timestamps stored (up to TIMING_WINDOW_SIZE) */
    int token_time_count;
} ThroughputTracker;

/**
 * @brief Initialize a throughput tracker
 *
 * Sets up the tracker and records the starting time.
 *
 * @param tracker Pointer to the ThroughputTracker structure to initialize
 */
void timing_init(ThroughputTracker *tracker);

/**
 * @brief Record a token generation event and calculate throughput
 *
 * Records the current timestamp and calculates both overall and windowed
 * throughput metrics.
 *
 * @param tracker Pointer to the ThroughputTracker structure
 * @param generated_tokens Total number of tokens generated so far (excluding input)
 * @param overall_tps Output parameter for overall tokens per second
 * @param window_tps Output parameter for windowed (last N tokens) tokens per second
 */
void timing_record_token(
    ThroughputTracker *tracker,
    int generated_tokens,
    float *overall_tps,
    float *window_tps);
