SRC_PATH   := src
BUILD_PATH := build
EXEC_NAME  := dotprod

KERNEL_SRC := $(SRC_PATH)/dotprod.cpp
TOP_FUNC   := DotProd
PLATFORM   := xilinx_u280_xdma_201920_3
XILINX_OBJ := $(BUILD_PATH)/$(TOP_FUNC).xo
CONNECT    := hbm_config.ini

EXECUTABLE := $(BUILD_PATH)/$(EXEC_NAME)

SRCS := $(shell find $(SRC_PATH) -name '*.c*' -o -name '*.h*')

CXX ?= g++
CXXFLAGS := -std=c++17
LDLIBS := \
    -L/lib/x86_64-linux-gpu -L/usr/local/lib \
    -ltapa -lfrt -lglog -lgflags -lOpenCL -lm

INCLUDES := \
	-I$(XILINX_HLS)/include/

.PHONY: all
all: run

.PHONY: build
build: $(EXECUTABLE)

$(EXECUTABLE): $(SRCS)
	mkdir -p $(BUILD_PATH);
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $(EXECUTABLE) $(LDLIBS)

.PHONY: kernel
kernel:
	tapac \
		-o build/solver.$(PLATFORM).hw.xo \
		--platform $(PLATFORM) \
		--top $(TOP_FUNC) \
		--work-dir build/solver.$(PLATFORM).hw.xo.tapa \
		--connectivity $(CONNECT) \
		--enable-hbm-binding-adjustment \
		--enable-synth-util \
		--run-floorplan-dse \
		--max-parallel-synth-jobs 16 \
		--floorplan-output build/solver.tcl \
		$(KERNEL_SRC)

.PHONY: run
run: $(EXECUTABLE)
	./$(EXECUTABLE)

.PHONY: clean
clean:
	rm -rf build/ work.out/
