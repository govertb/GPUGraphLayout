CC:=gcc

# Directories
L_SRC_DIR         := ../../lib
SRC_DIR           := ../../src
OBJ_DIR           := obj

# Compile with CUDA support by default
CUDA_SUPPORT ?= 1
ifeq ($(CUDA_SUPPORT), 1)
	CC:=nvcc
endif

# Debug compilation, disable by default
DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CXXFLAGS:=-g
	NVCCFLAGS:=-G
	O_LVL:=-O0
else
	NVCCFLAGS:=-D NDEBUG
	O_LVL:=-O3
endif

# C preprocessor flags
CPPFLAGS :=-D NO_FREETYPE

# C/C++ compiler flags
CFLAGS   :=$(O_LVL)
CXXFLAGS :=$(CXXFLAGS) $(O_LVL) -std=c++11

# Linker flags
LDFLAGS  :=-lc -lm -lstdc++ -lpng

# src./obj. files
GRAPH_VIEWER_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
GRAPH_VIEWER_OBJS := $(GRAPH_VIEWER_SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

PNGWRITER_SRCS    := $(L_SRC_DIR)/pngwriter/src/pngwriter.cc
PNGWRITER_OBJS    := $(PNGWRITER_SRCS:$(L_SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)

CUDA_SRCS         := $(wildcard $(SRC_DIR)/*.cu)
CUDA_OBJS         := $(CUDA_SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CUDA_DEPS         := $(wildcard $(SRC_DIR)/*.cuh)

CPP_SRC           := $(GRAPH_VIEWER_SRCS) $(PNGWRITER_SRCS)
CUDA_SRC          := $(CUDA_SRCS)
SOURCES           := $(C_SRC) $(CPP_SRC)
OBJECTS           := $(GRAPH_VIEWER_OBJS) $(PNGWRITER_OBJS)

ifeq ($(CUDA_SUPPORT), 1)
OBJECTS           := $(OBJECTS) $(CUDA_OBJS)
SOURCES           := $(SOURCES) $(CUDA_SRCS)
endif

# Generate dependency (.h, .hpp)
# ala http://stackoverflow.com/questions/2394609/makefile-header-dependencies
depend: .depend
.depend: $(SRCS)
	rm -f .depend
	$(CC) $(CXXFLAGS) $(CPPFLAGS) -M $(CPP_SRC) >> .depend

include ./.depend

graph_viewer: $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o graph_viewer

$(GRAPH_VIEWER_OBJS): $(GRAPH_VIEWER_SRCS)
	mkdir -p $(@D)
	$(CC) -c $(CPPFLAGS) $(CXXFLAGS) -o $@ $(@:$(OBJ_DIR)/%.o=$(SRC_DIR)/%.cpp)

$(CUDA_OBJS): $(CUDA_SRCS) $(CUDA_DEPS)
	mkdir -p $(@D)
	nvcc --device-c $(CXXFLAGS) $(NVCCFLAGS) $(CPPFLAGS) -o $@ $(@:$(OBJ_DIR)/%.o=$(SRC_DIR)/%.cu)

$(PNGWRITER_OBJS): $(PNGWRITER_SRCS)
	mkdir -p $(@D)
	$(CC) -c $(CPPFLAGS) $(CXXFLAGS) -o $@ $(@:$(OBJ_DIR)/%.o=$(L_SRC_DIR)/%.cc)

clear: clean

clean:
	rm -r graph_viewer $(OBJ_DIR)/* ./.depend

.PHONY: all clear clean depend
