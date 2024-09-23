#include <cstdint>
#include <tapa.h>
#include <ap_utils.h>

#define ACC_LATENCY 4

void Multiply(tapa::istream<float>& v1_q, tapa::istream<float>& v2_q,
         tapa::ostream<float>& prod_q, uint64_t n) {
Loop_Multiply: for (uint64_t i = 0; i < n; ++i) {
        prod_q << (v1_q.read() * v2_q.read());
    }
}

void Accumulate(tapa::istream<float>& prod_q,
         tapa::mmap<float> prod_out, uint64_t n, tapa::ostream<bool>& stop_flag) {
    float bins[ACC_LATENCY];
#pragma HLS array_partition variable = bins complete dim = 1
    float sum = 0;

Loop_Accumulate_Binning: for (uint64_t i = 0; i < n; ++i) {
#pragma HLS pipeline II=4
#pragma HLS unroll factor=4
        bins[i % ACC_LATENCY] += prod_q.read();
    }

Loop_Accumulate_Join: for (int i = 0; i < ACC_LATENCY; ++i) {
    #pragma HLS unroll
        sum += bins[i];
    }

    prod_out[0] = sum;
    stop_flag << true;
}

void Mmap2Stream(tapa::mmap<const float> mmap, uint64_t n,
                 tapa::ostream<float>& stream) {
Loop_Mmap2Stream: for (uint64_t i = 0; i < n; ++i) {
        stream << mmap[i];
    }
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

void DotProd(tapa::mmap<const float> v1, tapa::mmap<const float> v2,
             tapa::mmap<float> prod_out, uint64_t n, tapa::mmap<uint64_t> total) {
    tapa::stream<float> v1_q("v1");
    tapa::stream<float> v2_q("v2");
    tapa::stream<float> prod_q("prod");

    tapa::stream<bool> stop_flag("stop");

    tapa::task()
        .invoke(counter, total, stop_flag)
        .invoke(Mmap2Stream, v1, n, v1_q)
        .invoke(Mmap2Stream, v2, n, v2_q)
        .invoke(Multiply, v1_q, v2_q, prod_q, n)
        .invoke(Accumulate, prod_q, prod_out, n, stop_flag);
}
