#	-- graph-search-gpu --
#
#	graph-search-gpu's project Makefile.
#
#	Utilization example:
#		make <TARGET> ["DEBUG=true"]
#
#	@param TARGET
#		Can be any of the following:
#		all - builds the project (DEFAULT TARGET)
#		clean - cleans up all binaries generated during compilation
#		redo - cleans up and then builds
#
#	@param "DEBUG=true"
#		When present, the build will happen in debug mode.
#
#	@author
#		@hcpsilva - Henrique Silva
#
#	Make's default action is "all" when no parameters are provided.


################################################################################
#	Definitions:

#	- Project's directories:
INC_DIR := include
OBJ_DIR := bin
OUT_DIR := build
SRC_DIR := src
LIB_DIR := lib

#	Add the extra paths through these variables in the command line
LIB_EXTRA :=
INC_EXTRA :=

#	Default CUDA path, define it through the shell to something else if needed
CUDA_PATH ?= /opt/cuda

#	- Compilation flags:
#	Compiler and language version
CC := g++ -std=c++17
CUDAC ?= clang++ -std=c++17 -x cuda --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=sm_35
#	If DEBUG is defined (through command line), we'll turn on the debug flag and
#	attach address sanitizer on the executables.
DEBUGF := $(if $(DEBUG),-g -fsanitize=address -fno-omit-frame-pointer)
CFLAGS :=\
	-Wall \
	-Wextra \
	-Wpedantic \
	-Wshadow \
	-Wunreachable-code
OPT := $(if $(DEBUG),-O0,-O3 -march=native)
LIB := -L$(LIB_DIR) -L$(CUDA_PATH)/lib64 $(LIB_EXTRA) \
	-lfmt \
	-lcudart \
	-lcudadevrt
INC := -I$(INC_DIR) -I$(SRC_DIR) $(INC_EXTRA)

################################################################################
#	Files:

#	- Main source files:
#	Presumes that all "main" source files are in the root of SRC_DIR
MAIN := $(wildcard $(SRC_DIR)/*.cpp)

#	- Path to all final binaries:
TARGET := $(patsubst %.cpp, $(OUT_DIR)/%, $(notdir $(MAIN)))

#	- Other source files:
SRC := $(shell find $(SRC_DIR) -mindepth 2 -name '*.cpp' | cut -d'/' -f2-)

#	- Objects to be created:
OBJ := $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(SRC))

#	- CUDA objects:
CUDA := $(patsubst %.cu, $(OBJ_DIR)/%.o, $(shell find $(SRC_DIR) -mindepth 2 -name '*.cu' | cut -d'/' -f2-))

################################################################################
#	Rules:

#	- Executables:
$(TARGET): $(OUT_DIR)/%: $(SRC_DIR)/%.cpp $(OBJ) $(CUDA)
	$(CC) -o $@ $^ $(INC) $(LIB) $(DEBUGF) $(OPT)

#	- Objects:
$(OBJ): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(INC) $(CFLAGS) $(DEBUGF) $(OPT)

#	- CUDA objects:
$(CUDA): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(CUDAC) -c -o $@ $< $(INC)

################################################################################
#	Targets:

.DEFAULT_GOAL = all

all: $(TARGET)

clean:
	rm -rf $(OBJ_DIR)/* $(INC_DIR)/*~ $(TARGET) *~ *.o

redo: clean all

################################################################################
#	Debugging and etc.:

#	Debug of the Make variables
print-%:
	@echo $* = $($*)

.PHONY: all clean redo print-%
