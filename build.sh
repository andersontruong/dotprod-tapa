#!/bin/sh

g++ \
    -std=c++17 \
    src/*.cpp \
    -o dotprod \
    -I${HOME}/.rapidstream-tapa/usr/include/ \
    -I/opt/tools/xilinx/Vitis_HLS/2024.1/include/ \
    -Wl,-rpath,$(readlink -f ~/.rapidstream-tapa/usr/lib) \
    -L ${HOME}/.rapidstream-tapa/usr/lib/ \
    -ltapa -lfrt -lglog -lgflags -l:libOpenCL.so.1 -ltinyxml2 -lstdc++fs
