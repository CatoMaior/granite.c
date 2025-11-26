/**
 * @file console_utils.c
 * @brief Implementation of console output utilities
 */

#include "console_utils.h"
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

int console_get_terminal_width(void) {
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) {
        return w.ws_col;
    }
    return 80; // Default fallback width
}

int console_calculate_visual_height(const char *prefix, const char *text, int term_width) {
    int lines = 1;
    int col = 0;

    // Count prefix length
    int prefix_len = strlen(prefix);
    col += prefix_len;

    // If prefix is already longer than width (rare case), handle wrap
    while (col >= term_width) {
        lines++;
        col -= term_width;
    }

    // Now we scan the text character by character
    for (int i = 0; text[i] != '\0'; i++) {
        if (text[i] == '\n') {
            // Explicit line break: new line, column resets to 0
            lines++;
            col = 0;
        } else {
            // Normal character
            col++;
            // If we exceed the width, the terminal wraps automatically
            if (col >= term_width) {
                lines++;
                col = 0;
            }
        }
    }
    return lines;
}

void console_clear_previous_stream(StreamState *state) {
    if (state->last_lines > 0) {
        for (int i = 0; i < state->last_lines; i++) {
            // Go up one line and clear it
            // \033[A = Up, \033[2K = Clear Line
            if (i > 0) printf("\033[A");
            printf("\r\033[2K");
        }
        // One last \r for safety
        printf("\r");
    } else {
        // Base case: just clear the current line
        printf("\r\033[2K");
    }
    fflush(stdout);
}

void console_update_stream_state(StreamState *state, const char *prefix, const char *text, int term_width) {
    state->last_lines = console_calculate_visual_height(prefix, text, term_width);
}
