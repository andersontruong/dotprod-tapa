#include <cstdint>
#include <tapa.h>
#include <ap_utils.h>

#define N_BINS 8

void MAC(tapa::istream<float>& v1_q, tapa::istream<float>& v2_q, tapa::mmap<float> dp_out, tapa::ostream<bool>& stop_flag, uint64_t n) {
    float bins[N_BINS] = { 0.0 };
    float bins_1[N_BINS] = { 0.0 };
    #pragma HLS array_partition variable=bins type=complete
    float total = 0;

Loop_MAC: for (uint64_t i = 0; i < n; ++i) {
        #pragma HLS loop_tripcount min=1000 max=1000
        #pragma HLS dependence variable=bins type=inter distance=8 true
        #pragma HLS dependence variable=bins_1 type=inter distance=8 true
        #pragma HLS pipeline II=1
        bins[i % N_BINS] = v1_q.read() * v2_q.read();
        if (i > 0) {
            bins_1[(i - 1) % N_BINS] += bins[(i - 1) % N_BINS];
        }
    }
    bins_1[(n - 1) % N_BINS] += bins[(n - 1) % N_BINS];

Loop_Combine: for (uint64_t i = 0; i < N_BINS; ++i) {
        #pragma HLS unroll
        total += bins_1[i];
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

void Mmap2Stream(tapa::mmap<const float> mmap, uint64_t n,
                 tapa::ostream<float>& stream) {
Loop_Mmap2Stream: for (uint64_t i = 0; i < n; ++i) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=1000 max=1000
        stream << mmap[i];
    }
}

void DotProd(tapa::mmap<const float> v1, tapa::mmap<const float> v2,
             tapa::mmap<float> prod_out, uint64_t n, tapa::mmap<uint64_t> total) {
    tapa::stream<float> v1_q("v1");
    tapa::stream<float> v2_q("v2");

    tapa::stream<bool> stop_flag("stop");

    tapa::task()
        .invoke(counter, total, stop_flag)
        .invoke(Mmap2Stream, v1, n, v1_q)
        .invoke(Mmap2Stream, v2, n, v2_q)
        .invoke(MAC, v1_q, v2_q, prod_out, stop_flag, n)
    ;
}
