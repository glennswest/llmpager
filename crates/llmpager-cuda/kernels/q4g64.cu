// q4g64 GEMV: y = W x for a q4g64-quantized W [rows, cols].
//
// Blob layout (matches llmpager-core::quant): all f16 scales first
// (rows * cols/64, row-major), then nibble data (rows * cols/2); value i of
// a group lives in byte i/2, even i in the low nibble, stored as q+8.
//
// One warp per output row. Each lane owns whole groups (strided by 32), so
// scale and nibble reads are coalesced across the warp within a row and the
// fp32 x vector is re-read from L2/L1 (it is tiny next to W).
//
// Correctness-first: vectorized loads and half2 math come later (M3).

#include <cuda_fp16.h>

extern "C" __global__ void q4g64_gemv(
    const unsigned char* __restrict__ blob,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols)
{
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;

    const int groups = cols >> 6;
    const unsigned char* scales = blob;
    const unsigned char* data = blob + (size_t)rows * groups * 2;

    float acc = 0.f;
    for (int g = lane; g < groups; g += 32) {
        const __half s = *reinterpret_cast<const __half*>(
            scales + ((size_t)row * groups + g) * 2);
        const float sf = __half2float(s);
        const unsigned char* dp = data + (size_t)row * (cols >> 1) + (size_t)g * 32;
        const float* xp = x + g * 64;
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const unsigned char b = dp[i];
            sum += (float)((int)(b & 0x0F) - 8) * xp[2 * i];
            sum += (float)((int)(b >> 4) - 8) * xp[2 * i + 1];
        }
        acc += sf * sum;
    }

#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    }
    if (lane == 0) y[row] = acc;
}
