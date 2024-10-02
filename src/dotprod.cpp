#include <cstdint>
#include <tapa.h>
#include <ap_int.h>

#define N_BINS 8

using float_vec16 = ap_uint<512>;

void AccumulateStream(tapa::istreams<float, 1>& in_stream, tapa::mmap<float> out_sum, uint64_t n, tapa::ostream<bool>& stop_flag) {
    float temp[N_BINS] = { 0.0 };
    #pragma HLS array_partition variable=temp type=complete
    float acc_bins[N_BINS] = { 0.0 };
    float total = 0.0;

    for (uint64_t i = 0; i < n; ++i) {
        #pragma HLS loop_tripcount min=1000 max=1000
        #pragma HLS dependence variable=temp type=inter distance=8 true
        #pragma HLS dependence variable=acc_bins type=inter distance=8 true
        #pragma HLS pipeline II=1
        temp[i % N_BINS] = in_stream[0].read();
        if (i > 0) {
            acc_bins[(i - 1) % N_BINS] += temp[(i - 1) % N_BINS];
        }
    }
    acc_bins[(n - 1) % N_BINS] += temp[(n - 1) % N_BINS];

    for (int i = 0; i < N_BINS; ++i) {
        #pragma HLS unroll
        total += acc_bins[i];
    }
    out_sum[0] = total;
    stop_flag << true;
}

void AccumulateReduceStream_16x8(tapa::istreams<float, 16>& in_streams, tapa::ostreams<float, 8>& out_streams, uint64_t n) {
    Loop_Accumulate_Reduce: for (uint64_t i = 0; i < n; ++i) {
        #pragma HLS pipeline II=1
        Loop_Accumulate_Reduce_Inner: for (int j = 0; j < 8; ++j) {
            float t1 = in_streams[j*2].read();
            float t2 = in_streams[(j*2)+1].read();
            float sum = t1 + t2;
            out_streams[j] << sum;
        }
    }
}

void AccumulateReduceStream_8x4(tapa::istreams<float, 8>& in_streams, tapa::ostreams<float, 4>& out_streams, uint64_t n) {
    Loop_Accumulate_Reduce: for (uint64_t i = 0; i < n; ++i) {
        #pragma HLS pipeline II=1
        Loop_Accumulate_Reduce_Inner: for (int j = 0; j < 4; ++j) {
            float t1 = in_streams[j*2].read();
            float t2 = in_streams[(j*2)+1].read();
            float sum = t1 + t2;
            out_streams[j] << sum;
        }
    }
}

void AccumulateReduceStream_4x2(tapa::istreams<float, 4>& in_streams, tapa::ostreams<float, 2>& out_streams, uint64_t n) {
    Loop_Accumulate_Reduce: for (uint64_t i = 0; i < n; ++i) {
        #pragma HLS pipeline II=1
        Loop_Accumulate_Reduce_Inner: for (int j = 0; j < 2; ++j) {
            float t1 = in_streams[j*2].read();
            float t2 = in_streams[(j*2)+1].read();
            float sum = t1 + t2;
            out_streams[j] << sum;
        }
    }
}

void AccumulateReduceStream_2x1(tapa::istreams<float, 2>& in_streams, tapa::ostreams<float, 1>& out_streams, uint64_t n) {
    Loop_Accumulate_Reduce: for (uint64_t i = 0; i < n; ++i) {
        #pragma HLS pipeline II=1
        Loop_Accumulate_Reduce_Inner: for (int j = 0; j < 1; ++j) {
            float t1 = in_streams[j*2].read();
            float t2 = in_streams[(j*2)+1].read();
            float sum = t1 + t2;
            out_streams[j] << sum;
        }
    }
}

void Multiply(tapa::istream<float_vec16>& v1_q, tapa::istream<float_vec16>& v2_q, tapa::ostreams<float, 16>& prod_q, uint64_t n) {
    Loop_MAC: for (uint64_t i = 0; i < n; ++i) {
        #pragma HLS loop_tripcount min=1000 max=1000
        #pragma HLS pipeline II=1
        float_vec16 v1 = v1_q.read();
        float_vec16 v2 = v2_q.read();
        Loop_MAC_Inner: for (int j = 0; j < 16; ++j) {
            #pragma HLS unroll
            unsigned int v1_uint = v1.range(32*(j+1) - 1, 32*j);
            unsigned int v2_uint = v2.range(32*(j+1) - 1, 32*j);
            float v1_float = *((float*)(&v1_uint));
            float v2_float = *((float*)(&v2_uint));
            float prod = v1_float * v2_float;
            prod_q[j] << prod;
        }
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

    tapa::streams<float, 16> s_16;
    tapa::streams<float, 8> s_8;
    tapa::streams<float, 4> s_4;
    tapa::streams<float, 2> s_2;
    tapa::streams<float, 1> s_1;

    tapa::task()
        .invoke(counter, total, stop_flag)
        .invoke(Mmap2Stream, v1, n, v1_q)
        .invoke(Mmap2Stream, v2, n, v2_q)
        .invoke(Multiply, v1_q, v2_q, s_16, n)
        .invoke(AccumulateReduceStream_16x8, s_16, s_8, n)
        .invoke(AccumulateReduceStream_8x4, s_8, s_4, n)
        .invoke(AccumulateReduceStream_4x2, s_4, s_2, n)
        .invoke(AccumulateReduceStream_2x1, s_2, s_1, n)
        .invoke(AccumulateStream, s_1, prod_out, n, stop_flag)
    ;
}
