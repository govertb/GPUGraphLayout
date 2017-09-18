CC:=gcc

# Directories
L_SRC_DIR         := ../../lib
SRC_DIR           := ../../src
OBJ_DIR           := obj

# C/C++ compiler flags
COMMON_FLAGS:=-O3
CFLAGS   :=$(COMMON_FLAGS) -D NO_FREETYPE
CXXFLAGS :=$(COMMON_FLAGS) -std=c++11 #-g
NVCCFLAGS:=#-G

# C preprocessor flags
CPPFLAGS :=-D NO_FREETYPE
ifeq ($(CUDA_SUPPORT), 1)
	CPPFLAGS:=$(CPPFLAGS) -D __CUDA__ -isystem /usr/local/cuda/include/
endif
ifeq ($(DEBUG_PRINTING), 1)
	CPPFLAGS:=$(CPPFLAGS) -D DEBUGPRINT
endif

# Linker flags
LDFLAGS  :=-lc -lm -lstdc++ -fopenmp -lpng
ifeq ($(CUDA_SUPPORT), 1)
	LDFLAGS:=$(LDFLAGS) -L/usr/local/cuda/lib64 -lcudart
endif

# src./obj. files


GRAPH_VIEWER_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
GRAPH_VIEWER_OBJS := $(GRAPH_VIEWER_SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

PNGWRITER_SRCS    := $(L_SRC_DIR)/pngwriter/src/pngwriter.cc
PNGWRITER_OBJS    := $(PNGWRITER_SRCS:$(L_SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)

CUDA_SRCS         := $(wildcard $(SRC_DIR)/*.cu)
CUDA_OBJS         := $(CUDA_SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CUDA_DEPS         := $(wildcard $(SRC_DIR)/*.cuh)

# Don't compile with CUDA support by default
CUDA_SUPPORT      ?= 0

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
	rm -f ./.depend
	$(CC) $(CXXFLAGS) $(CPPFLAGS) -MM $(CPP_SRC) >> ./.depend

include ./.depend

graph_viewer: $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o graph_viewer

$(GRAPH_VIEWER_OBJS): $(GRAPH_VIEWER_SRCS)
	mkdir -p $(@D)
	$(CC) -c $(CPPFLAGS) $(CXXFLAGS) --std=c++11 -o $@ $(@:$(OBJ_DIR)/%.o=$(SRC_DIR)/%.cpp)

$(CUDA_OBJS): $(CUDA_SRCS) $(CUDA_DEPS)
	mkdir -p $(@D)
	nvcc -c $(CXXFLAGS) $(NVCCFLAGS) $(CPPFLAGS) -o $@ $(@:$(OBJ_DIR)/%.o=$(SRC_DIR)/%.cu)

$(PNGWRITER_OBJS): $(PNGWRITER_SRCS)
	mkdir -p $(@D)
	$(CC) -c $(CPPFLAGS) $(CXXFLAGS) -o $@ $(@:$(OBJ_DIR)/%.o=$(L_SRC_DIR)/%.cc)

clear: clean

clean:
	rm -r graph_viewer $(OBJ_DIR)/* ./.depend

.PHONY: all clear clean depend
