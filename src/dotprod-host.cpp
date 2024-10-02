// Copyright (c) 2024 RapidStream Design Automation, Inc. and contributors.
// All rights reserved. The contributor(s) of this file has/have agreed to the
// RapidStream Contributor License Agreement.

#include <iostream>
#include <vector>
#include <array>
#include <ap_int.h>

#include <gflags/gflags.h>
#include <tapa.h>

using std::clog;
using std::endl;
using std::vector;
using std::array;
using float_vec16 = ap_uint<512>;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

void DotProd_CPU(aligned_vector<float> v1, aligned_vector<float> v2, float& output, uint64_t n) {
    float sum = 0;
    for (uint64_t i = 0; i < n; ++i) {
        sum += v1[i] * v2[i];
    }
    output = sum;
}

void DotProd(tapa::mmap<const float_vec16> v1, tapa::mmap<const float_vec16> v2,
             tapa::mmap<float> prod_out, uint64_t n, tapa::mmap<uint64_t> total);

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

    const uint64_t n = argc > 1 ? atoll(argv[1]) : 512 * 512;
    aligned_vector<float> v1(n);
    aligned_vector<float> v2(n);
    aligned_vector<float> output = { 0 };
    float& output_tapa = output[0];
    aligned_vector<uint64_t> total = { 0 };
    uint64_t& cycles = total[0];

    for (uint64_t i = 0; i < n; ++i) {
        v1[i] = static_cast<float>(i);
        v2[i] = static_cast<float>(i) * 2;
    }

    aligned_vector<float_vec16> v1_vec(n >> 4);
    aligned_vector<float_vec16> v2_vec(n >> 4);

    std::memcpy(v1_vec.data(), v1.data(), sizeof(float) * n);
    std::memcpy(v2_vec.data(), v2.data(), sizeof(float) * n);

    int64_t kernel_time_ns = tapa::invoke(
        DotProd,
        FLAGS_bitstream,
        tapa::read_only_mmap<const float_vec16>(v1_vec),
        tapa::read_only_mmap<const float_vec16>(v2_vec),
        tapa::write_only_mmap<float>(output),
        n >> 4,
        tapa::write_only_mmap<uint64_t>(total)
    );

    clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << endl;
    clog << "TAPA Output: " << output_tapa << endl;

    float output_cpu = 0;
    DotProd_CPU(v1, v2, output_cpu, n);
    clog << "CPU Output: " << output_cpu << endl;

    clog << "Vector size N=" << n << endl;
    clog << "Packed vector size = " << v1_vec.size() << endl;
    clog << "Total Cycles: " << cycles << endl;

    clog << "\nCycles/elem = " << (float)cycles / n << endl;

    if (output_tapa != output_cpu) {
        clog << "FAIL!" << endl;
        return 1;
    }

    return 0;
}
