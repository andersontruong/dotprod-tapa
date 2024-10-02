#include <cstdint>
#include <tapa.h>
#include <ap_int.h>

#define N_BINS 8

using float_vec16 = ap_uint<512>;

void MAC(tapa::istream<float_vec16>& v1_q, tapa::istream<float_vec16>& v2_q, tapa::mmap<float> dp_out, tapa::ostream<bool>& stop_flag, uint64_t n) {
    float bins[N_BINS][16] = { 0.0 };
    #pragma HLS array_partition variable=bins type=complete
    float bins_1[N_BINS][16] = { 0.0 };
    #pragma HLS array_partition variable=bins_1 type=complete
    float total = 0;

    Loop_MAC: for (uint64_t i = 0; i < n; ++i) {
        #pragma HLS loop_tripcount min=1000 max=1000
        #pragma HLS dependence variable=bins type=inter distance=8 true
        #pragma HLS dependence variable=bins_1 type=inter distance=8 true
        #pragma HLS pipeline II=1
        float_vec16 v1 = v1_q.read();
        float_vec16 v2 = v2_q.read();
        Loop_MAC_Inner: for (int j = 0; j < 16; ++j) {
            #pragma HLS unroll
            unsigned int v1_uint = v1.range(31, 0);
            unsigned int v2_uint = v2.range(31, 0);
            float v1_float = *((float*)(&v1_uint));
            float v2_float = *((float*)(&v2_uint));
            float prod = v1_float * v2_float;
            bins[i % N_BINS][j] = prod;
        }
        if (i > 0) {
            Loop_Acc_Column: for (int j = 0; j < 16; ++j) {
                #pragma HLS unroll
                bins_1[(i - 1) % N_BINS][j] += bins[(i - 1) % N_BINS][j];
            }
        }
    }
    Loop_Acc_Last_Column: for (int j = 0; j < 16; ++j) {
        #pragma HLS unroll
        bins_1[(n - 1) % N_BINS][j] += bins[(n - 1) % N_BINS][j];
    }

    Loop_Combine: for (uint64_t i = 0; i < N_BINS; ++i) {
        Loop_Combine_Inner: for (uint64_t j = 0; j < 16; ++j) {
            total += bins_1[i][j];
        }
    }

    dp_out[0] = total;
    stop_flag << true;
}

void counter(tapa::mmap<uint64_t> total, tapa::istream<bool>& stop_flag) {
    uint64_t count = 0;
    bool flag = false;
    #pragma HLS latency max=1
Loop_Counter: for (;;) {
        if (stop_flag.try_read(flag))
            break;
        count++;
    }
    total[0] = count;
}

void Mmap2Stream(tapa::mmap<const float_vec16> mmap, uint64_t n,
                 tapa::ostream<float_vec16>& stream) {
Loop_Mmap2Stream: for (uint64_t i = 0; i < n; ++i) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=1000 max=1000
        stream << mmap[i];
    }
}

void DotProd(tapa::mmap<const float_vec16> v1, tapa::mmap<const float_vec16> v2,
             tapa::mmap<float> prod_out, uint64_t n, tapa::mmap<uint64_t> total) {
    tapa::stream<float_vec16> v1_q("v1");
    tapa::stream<float_vec16> v2_q("v2");
    tapa::stream<bool> stop_flag("stop");

    tapa::task()
        .invoke(counter, total, stop_flag)
        .invoke(Mmap2Stream, v1, n, v1_q)
        .invoke(Mmap2Stream, v2, n, v2_q)
        .invoke(MAC, v1_q, v2_q, prod_out, stop_flag, n)
    ;
}
