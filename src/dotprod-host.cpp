// Copyright (c) 2024 RapidStream Design Automation, Inc. and contributors.
// All rights reserved. The contributor(s) of this file has/have agreed to the
// RapidStream Contributor License Agreement.

#include <iostream>
#include <vector>
#include <array>

#include <gflags/gflags.h>
#include <tapa.h>

using std::clog;
using std::endl;
using std::vector;
using std::array;

void DotProd_CPU(vector<float> v1, vector<float> v2, float& output, uint64_t n) {
    float sum = 0;
    for (uint64_t i = 0; i < n; ++i) {
        sum += v1[i] * v2[i];
    }
    output = sum;
}

void DotProd(tapa::mmap<const float> v1, tapa::mmap<const float> v2,
             tapa::mmap<float> output, uint64_t n);

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

    const uint64_t n = argc > 1 ? atoll(argv[1]) : 1024 * 1024;
    vector<float> v1(n);
    vector<float> v2(n);
    array<float, 1> output = { 0 };
    float& output_tapa = output[0];

    for (uint64_t i = 0; i < n; ++i) {
        v1[i] = static_cast<float>(i);
        v2[i] = static_cast<float>(i) * 2;
    }
    int64_t kernel_time_ns = tapa::invoke(
        DotProd,
        FLAGS_bitstream,
        tapa::read_only_mmap<const float>(v1),
        tapa::read_only_mmap<const float>(v2),
        tapa::write_only_mmap<float>(output),
        n
    );

    clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << endl;
    clog << "TAPA Output: " << output_tapa << endl;

    float output_cpu = 0;
    DotProd_CPU(v1, v2, output_cpu, n);
    clog << "CPU Output: " << output_cpu << endl;

    if (output_tapa != output_cpu) {
        clog << "FAIL!" << endl;
        return 1;
    }

    return 0;
}
