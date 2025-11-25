CC            = gcc
DEPFLAGS      = -MMD -MP
COMPILE_FLAGS = -Wall -Wextra -fopenmp
LINK_FLAGS    = -lm -fopenmp
FAST_FLAGS    = -Ofast -fno-finite-math-only
DEBUG_FLAGS   = -O0 -g

SRC_ROOT = $(wildcard *.c)
SRC_SRC  = $(wildcard src/*.c)
SRC      = $(SRC_ROOT) $(SRC_SRC)

# Release build directories and objects
RELEASE_DIR = build/release
RELEASE_OBJ_DIR = $(RELEASE_DIR)/obj
RELEASE_OUT = $(RELEASE_DIR)/granite-c
RELEASE_OBJ = $(patsubst %.c,$(RELEASE_OBJ_DIR)/%.o,$(notdir $(SRC)))
RELEASE_DEPS = $(RELEASE_OBJ:.o=.d)

# Debug build directories and objects
DEBUG_DIR = build/debug
DEBUG_OBJ_DIR = $(DEBUG_DIR)/obj
DEBUG_OUT = $(DEBUG_DIR)/granite-c
DEBUG_OBJ = $(patsubst %.c,$(DEBUG_OBJ_DIR)/%.o,$(notdir $(SRC)))
DEBUG_DEPS = $(DEBUG_OBJ:.o=.d)

# Default target: release build
all: $(RELEASE_OUT)

# Debug build
debug: $(DEBUG_OUT)

# Release binary linking
$(RELEASE_OUT): $(RELEASE_OBJ)
	@mkdir -p $(RELEASE_DIR)
	$(CC) $(RELEASE_OBJ) -o $(RELEASE_OUT) $(LINK_FLAGS)

# Debug binary linking
$(DEBUG_OUT): $(DEBUG_OBJ)
	@mkdir -p $(DEBUG_DIR)
	$(CC) $(DEBUG_OBJ) -o $(DEBUG_OUT) $(LINK_FLAGS)

# Release object compilation (root directory)
$(RELEASE_OBJ_DIR)/%.o: %.c Makefile
	@mkdir -p $(RELEASE_OBJ_DIR)
	$(CC) $(COMPILE_FLAGS) $(FAST_FLAGS) $(DEPFLAGS) -c $< -o $@

# Release object compilation (src directory)
$(RELEASE_OBJ_DIR)/%.o: src/%.c Makefile
	@mkdir -p $(RELEASE_OBJ_DIR)
	$(CC) $(COMPILE_FLAGS) $(FAST_FLAGS) $(DEPFLAGS) -c $< -o $@

# Debug object compilation (root directory)
$(DEBUG_OBJ_DIR)/%.o: %.c Makefile
	@mkdir -p $(DEBUG_OBJ_DIR)
	$(CC) $(COMPILE_FLAGS) $(DEBUG_FLAGS) $(DEPFLAGS) -c $< -o $@

# Debug object compilation (src directory)
$(DEBUG_OBJ_DIR)/%.o: src/%.c Makefile
	@mkdir -p $(DEBUG_OBJ_DIR)
	$(CC) $(COMPILE_FLAGS) $(DEBUG_FLAGS) $(DEPFLAGS) -c $< -o $@

run: all
	./$(RELEASE_OUT)

run-debug: debug
	./$(DEBUG_OUT)

clean:
	rm -rf build

-include $(RELEASE_DEPS)
-include $(DEBUG_DEPS)

.PHONY: all debug run run-debug clean