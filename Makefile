CC      = gcc
CFLAGS  = -Wall -Wextra -g -O0 -lm

BUILD_DIR = build
OBJ_DIR   = $(BUILD_DIR)/obj
OUT       = $(BUILD_DIR)/app

SRC_ROOT = $(wildcard *.c)
SRC_SRC  = $(wildcard src/*.c)
SRC      = $(SRC_ROOT) $(SRC_SRC)

OBJ      = $(patsubst %.c, $(OBJ_DIR)/%.o, $(notdir $(SRC)))

all: $(OUT)

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

.PHONY: all run clean
