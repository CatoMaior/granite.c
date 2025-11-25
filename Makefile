CC      = gcc
BASE_FLAGS  = -Wall -Wextra -lm -fopenmp -march=native
FAST_FLAGS  = -O3
DEBUG_FLAGS = -O0 -g

BUILD_DIR = build
OBJ_DIR   = $(BUILD_DIR)/obj
OUT       = $(BUILD_DIR)/granite-c

SRC_ROOT = $(wildcard *.c)
SRC_SRC  = $(wildcard src/*.c)
SRC      = $(SRC_ROOT) $(SRC_SRC)

OBJ      = $(patsubst %.c, $(OBJ_DIR)/%.o, $(notdir $(SRC)))

# Default target: optimized build
all: CFLAGS = $(BASE_FLAGS) $(FAST_FLAGS)
all: $(OUT)

# Debug target: debug build
debug: CFLAGS = $(BASE_FLAGS) $(DEBUG_FLAGS)
debug: clean $(OUT)

$(OUT): $(OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(OBJ) -o $(OUT)

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: src/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

run: $(OUT)
	./$(OUT)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all debug run clean
