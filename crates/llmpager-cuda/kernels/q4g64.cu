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

// Batched variant: E experts in one launch (grid.y = expert index).
// `blobs` holds E device base addresses; `region_off` selects gate/up/down
// within each blob. x advances by x_stride per expert (0 = shared input),
// y by `rows`. Same warp-per-row vectorized body as q4g64_gemv.
extern "C" __global__ void q4g64_gemv_batch(
    const unsigned long long* __restrict__ blobs,
    unsigned long long region_off,
    const float* __restrict__ x,
    int x_stride,
    float* __restrict__ y,
    int rows,
    int cols,
    int group)
{
    const int e = blockIdx.y;
    const unsigned char* blob = reinterpret_cast<const unsigned char*>(blobs[e] + region_off);
    const float* xe = x + (size_t)e * x_stride;
    float* ye = y + (size_t)e * rows;

    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;

    const int groups = cols / group;
    const int words = group >> 3;
    const unsigned char* scales = blob;
    const unsigned char* data = blob + (size_t)rows * groups * 2;

    float acc = 0.f;
    for (int g = lane; g < groups; g += 32) {
        const __half s = *reinterpret_cast<const __half*>(
            scales + ((size_t)row * groups + g) * 2);
        const float sf = __half2float(s);
        const unsigned int* dp = reinterpret_cast<const unsigned int*>(
            data + (size_t)row * (cols >> 1) + (size_t)g * (group >> 1));
        const float4* x4 = reinterpret_cast<const float4*>(xe + (size_t)g * group);
        float sum = 0.f;
        // fp16 magic-number unpack: OR nibbles into the fp16 mantissa at
        // exponent 1024 (0x6400), subtract 1032 (= 1024 + zero-point 8) in
        // half2 — two dequantized values per 3 ALU ops instead of ~10.
        // Pair layout within each word: (b & 0x000F000F) = (v0, v4), etc.
        const __half2 bias = __floats2half2_rn(1032.f, 1032.f);
#pragma unroll 8
        for (int i = 0; i < words; ++i) {
            const unsigned int b = dp[i];
            const float4 xa = x4[2 * i];
            const float4 xb = x4[2 * i + 1];
            unsigned int t0 = (b & 0x000F000Fu) | 0x64006400u;
            unsigned int t1 = ((b >> 4) & 0x000F000Fu) | 0x64006400u;
            unsigned int t2 = ((b >> 8) & 0x000F000Fu) | 0x64006400u;
            unsigned int t3 = ((b >> 12) & 0x000F000Fu) | 0x64006400u;
            const float2 v0 = __half22float2(__hsub2(*reinterpret_cast<__half2*>(&t0), bias));
            const float2 v1 = __half22float2(__hsub2(*reinterpret_cast<__half2*>(&t1), bias));
            const float2 v2 = __half22float2(__hsub2(*reinterpret_cast<__half2*>(&t2), bias));
            const float2 v3 = __half22float2(__hsub2(*reinterpret_cast<__half2*>(&t3), bias));
            sum += v0.x * xa.x + v0.y * xb.x;
            sum += v1.x * xa.y + v1.y * xb.y;
            sum += v2.x * xa.z + v2.y * xb.z;
            sum += v3.x * xa.w + v3.y * xb.w;
        }
        acc += sf * sum;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) ye[row] = acc;
}

// out[i] += sum_e wts[e] * eouts[e, i]
extern "C" __global__ void moe_reduce_f32(
    const float* __restrict__ eouts,
    const float* __restrict__ wts,
    float* __restrict__ out,
    int experts,
    int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float a = 0.f;
        for (int e = 0; e < experts; ++e) {
            a += wts[e] * eouts[(size_t)e * n + i];
        }
        out[i] += a;
    }
}

extern "C" __global__ void q4g64_gemv(
    const unsigned char* __restrict__ blob,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols,
    int group)
{
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;

    const int groups = cols / group;
    const int words = group >> 3; // uint words of nibble data per group
    const unsigned char* scales = blob;
    const unsigned char* data = blob + (size_t)rows * groups * 2;

    float acc = 0.f;
    for (int g = lane; g < groups; g += 32) {
        const __half s = *reinterpret_cast<const __half*>(
            scales + ((size_t)row * groups + g) * 2);
        const float sf = __half2float(s);
        // group/2 data bytes per group as uint loads; x as float4 pairs.
        // (Data region offset stays 4-byte aligned for any group %8==0.)
        const unsigned int* dp = reinterpret_cast<const unsigned int*>(
            data + (size_t)row * (cols >> 1) + (size_t)g * (group >> 1));
        const float4* x4 = reinterpret_cast<const float4*>(x + (size_t)g * group);
        float sum = 0.f;
        // fp16 magic-number unpack: OR nibbles into the fp16 mantissa at
        // exponent 1024 (0x6400), subtract 1032 (= 1024 + zero-point 8) in
        // half2 — two dequantized values per 3 ALU ops instead of ~10.
        // Pair layout within each word: (b & 0x000F000F) = (v0, v4), etc.
        const __half2 bias = __floats2half2_rn(1032.f, 1032.f);
#pragma unroll 8
        for (int i = 0; i < words; ++i) {
            const unsigned int b = dp[i];
            const float4 xa = x4[2 * i];
            const float4 xb = x4[2 * i + 1];
            unsigned int t0 = (b & 0x000F000Fu) | 0x64006400u;
            unsigned int t1 = ((b >> 4) & 0x000F000Fu) | 0x64006400u;
            unsigned int t2 = ((b >> 8) & 0x000F000Fu) | 0x64006400u;
            unsigned int t3 = ((b >> 12) & 0x000F000Fu) | 0x64006400u;
            const float2 v0 = __half22float2(__hsub2(*reinterpret_cast<__half2*>(&t0), bias));
            const float2 v1 = __half22float2(__hsub2(*reinterpret_cast<__half2*>(&t1), bias));
            const float2 v2 = __half22float2(__hsub2(*reinterpret_cast<__half2*>(&t2), bias));
            const float2 v3 = __half22float2(__hsub2(*reinterpret_cast<__half2*>(&t3), bias));
            sum += v0.x * xa.x + v0.y * xb.x;
            sum += v1.x * xa.y + v1.y * xb.y;
            sum += v2.x * xa.z + v2.y * xb.z;
            sum += v3.x * xa.w + v3.y * xb.w;
        }
        acc += sf * sum;
    }

#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    }
    if (lane == 0) y[row] = acc;
}
