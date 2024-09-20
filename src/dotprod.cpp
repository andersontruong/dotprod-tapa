#include <cstdint>

#include <tapa.h>

void Multiply(tapa::istream<float>& v1_q, tapa::istream<float>& v2_q,
         tapa::ostream<float>& prod_q, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) {
        prod_q << (v1_q.read() * v2_q.read());
    }
}

void Accumulate(tapa::istream<float>& prod_q,
         tapa::mmap<float> output, uint64_t n) {
    float sum = 0;
    for (uint64_t i = 0; i < n; ++i) {
        sum += prod_q.read();
    }
    output[0] = sum;
}

void Mmap2Stream(tapa::mmap<const float> mmap, uint64_t n,
                 tapa::ostream<float>& stream) {
    for (uint64_t i = 0; i < n; ++i) {
        stream << mmap[i];
    }
}

void DotProd(tapa::mmap<const float> v1, tapa::mmap<const float> v2,
             tapa::mmap<float> output, uint64_t n) {
    tapa::stream<float> v1_q("v1");
    tapa::stream<float> v2_q("v2");
    tapa::stream<float> prod_q("prod");

    tapa::task()
        .invoke(Mmap2Stream, v1, n, v1_q)
        .invoke(Mmap2Stream, v2, n, v2_q)
        .invoke(Multiply, v1_q, v2_q, prod_q, n)
        .invoke(Accumulate, prod_q, output, n);
}
