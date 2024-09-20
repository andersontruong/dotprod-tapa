
SRC_PATH := src
BUILD_PATH := build
EXEC_NAME := dotprod

EXECUTABLE := $(BUILD_PATH)/$(EXEC_NAME)

SRCS := $(shell find $(SRC_PATH) -name '*.c*' -o -name '*.h*')

CXX ?= g++
CXXFLAGS := -std=c++17
LDLIBS := \
	-Wl,-rpath,$(shell readlink -f ~/.rapidstream-tapa/usr/lib) \
    -L ${HOME}/.rapidstream-tapa/usr/lib/ \
    -ltapa -lfrt -lglog -lgflags -l:libOpenCL.so.1 -ltinyxml2 -lstdc++fs

INCLUDES := \
	-I${HOME}/.rapidstream-tapa/usr/include/\
	-I/opt/tools/xilinx/Vitis_HLS/2024.1/include/

.PHONY: all
all: run

.PHONY: build
build: $(EXECUTABLE)

$(EXECUTABLE): $(SRCS)
	mkdir -p $(BUILD_PATH);
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $(EXECUTABLE) $(LDLIBS)

.PHONY: run
run: $(EXECUTABLE)
	./$(EXECUTABLE)

.PHONY: clean
clean:
	rm -rf build/
