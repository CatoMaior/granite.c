/**
 * @file console_utils.h
 * @brief Console output utilities for streaming text display
 *
 * This file provides utilities for managing terminal output, including
 * calculating visual height and clearing previous output for streaming display.
 */

#pragma once

/**
 * @struct StreamState
 * @brief Structure for tracking streaming output state
 *
 * This structure maintains state needed for clearing and updating
 * streaming text output in the terminal.
 */
typedef struct {
    /** @brief Number of lines occupied by last stream output */
    int last_lines;
} StreamState;

/**
 * @brief Get the current terminal width in columns
 *
 * Queries the terminal to determine its width. Falls back to 80 columns
 * if the query fails.
 *
 * @return Terminal width in columns
 */
int console_get_terminal_width(void);

/**
 * @brief Calculate the visual height of text when displayed in terminal
 *
 * Calculates how many lines the text will occupy in the terminal,
 * accounting for line wrapping based on terminal width. Considers
 * both explicit newlines and automatic wrapping.
 *
 * @param prefix Text prefix (e.g., prompt) that appears before the text
 * @param text Main text content to measure
 * @param term_width Terminal width in columns
 * @return Number of lines the text will occupy
 */
int console_calculate_visual_height(const char *prefix, const char *text, int term_width);

/**
 * @brief Clear the previous streaming output from terminal
 *
 * Uses ANSI escape codes to move cursor up and clear lines that
 * were occupied by the previous streaming output.
 *
 * @param state Pointer to StreamState containing line count to clear
 */
void console_clear_previous_stream(StreamState *state);

/**
 * @brief Update streaming output state
 *
 * Calculates and stores the visual height for the next clear operation.
 *
 * @param state Pointer to StreamState to update
 * @param prefix Text prefix (e.g., prompt)
 * @param text Current text content
 * @param term_width Terminal width in columns
 */
void console_update_stream_state(StreamState *state, const char *prefix, const char *text, int term_width);
