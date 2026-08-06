// Batch-1 decode kernels. Correctness-first; fusion and fp16 KV are M3.
//
// Activations are f32 throughout. Core weights arrive bf16 (as stored in the
// checkpoint); norm weights are pre-converted to f32 at load time.

#include <cuda_fp16.h>

static __device__ __forceinline__ float bf16_to_f32(unsigned short h) {
    unsigned int b = ((unsigned int)h) << 16;
    return __uint_as_float(b);
}

// y = x * rsqrt(mean(x^2) + eps) * w        (single block)
extern "C" __global__ void rmsnorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int n,
    float eps)
{
    __shared__ float red[256];
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        acc += x[i] * x[i];
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        y[i] = x[i] * inv * w[i];
    }
}

// y[rows] = W x, W bf16 [rows, cols] row-major. One warp per row.
extern "C" __global__ void bf16_gemv(
    const unsigned short* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols)
{
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const unsigned short* wr = w + (size_t)row * cols;
    float acc = 0.f;
    for (int c = lane; c < cols; c += 32) {
        acc += bf16_to_f32(wr[c]) * x[c];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

// out = silu(gate) * up
extern "C" __global__ void silu_mul_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const float g = gate[i];
        out[i] = g / (1.f + __expf(-g)) * up[i];
    }
}

// a += b
extern "C" __global__ void add_f32(
    float* __restrict__ a,
    const float* __restrict__ b,
    int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        a[i] += b[i];
    }
}

// NeoX rotate-half RoPE, applied in place to [heads, head_dim] at `pos`.
// Pair (i, i + hd/2): x1' = x1 c - x2 s ; x2' = x1 s + x2 c, with
// theta_i = pos * base^(-2i/hd).
extern "C" __global__ void rope_f32(
    float* __restrict__ x,
    int heads,
    int head_dim,
    int pos,
    float base)
{
    const int half = head_dim >> 1;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < heads * half;
         idx += gridDim.x * blockDim.x) {
        const int h = idx / half;
        const int i = idx % half;
        const float freq = __powf(base, -2.f * (float)i / (float)head_dim);
        const float ang = (float)pos * freq;
        const float c = __cosf(ang);
        const float s = __sinf(ang);
        float* p = x + (size_t)h * head_dim;
        const float x1 = p[i];
        const float x2 = p[i + half];
        p[i] = x1 * c - x2 * s;
        p[i + half] = x1 * s + x2 * c;
    }
}

// Decode attention, one block per query head. K/V caches are f32
// [kv_heads, max_seq, head_dim]; `scratch` is [heads, max_seq] global
// workspace for softmax weights. GQA maps head -> head / (heads/kv_heads).
extern "C" __global__ void attn_decode_f32(
    const float* __restrict__ q,       // [heads, head_dim]
    const float* __restrict__ kcache,  // [kv_heads, max_seq, head_dim]
    const float* __restrict__ vcache,
    float* __restrict__ out,           // [heads, head_dim]
    float* __restrict__ scratch,       // [heads, max_seq]
    int heads,
    int kv_heads,
    int head_dim,
    int seq_len,
    int max_seq,
    float scale)
{
    const int h = blockIdx.x;
    if (h >= heads) return;
    const int kv = h / (heads / kv_heads);
    const float* qh = q + (size_t)h * head_dim;
    const float* kh = kcache + (size_t)kv * max_seq * head_dim;
    const float* vh = vcache + (size_t)kv * max_seq * head_dim;
    float* sc = scratch + (size_t)h * max_seq;
    __shared__ float red[256];

    // Scores.
    float lmax = -1e30f;
    for (int p = threadIdx.x; p < seq_len; p += blockDim.x) {
        const float* kp = kh + (size_t)p * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kp[d];
        const float s = dot * scale;
        sc[p] = s;
        lmax = fmaxf(lmax, s);
    }
    red[threadIdx.x] = lmax;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    const float m = red[0];
    __syncthreads();

    // Softmax weights (unnormalized) + partition sum.
    float lsum = 0.f;
    for (int p = threadIdx.x; p < seq_len; p += blockDim.x) {
        const float e = __expf(sc[p] - m);
        sc[p] = e;
        lsum += e;
    }
    red[threadIdx.x] = lsum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float invz = 1.f / red[0];
    __syncthreads();

    // Weighted V sum; threads split output dims.
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int p = 0; p < seq_len; ++p) {
            acc += sc[p] * vh[(size_t)p * head_dim + d];
        }
        out[(size_t)h * head_dim + d] = acc * invz;
    }
}

// Copy one bf16 row (e.g. an embedding) to f32.
extern "C" __global__ void bf16_row_to_f32(
    const unsigned short* __restrict__ table,
    int row,
    int n,
    float* __restrict__ out)
{
    const unsigned short* src = table + (size_t)row * n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = bf16_to_f32(src[i]);
    }
}
