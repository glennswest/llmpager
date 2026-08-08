// MLA (multi-head latent attention) decode kernels — DeepSeek-V3 family
// (Kimi K2.x). Absorbed decode path:
//
//   cache row (per token) = [ c_kv_norm (c_dim) | k_rope (r_dim) ]
//   query    (per head)   = [ q_eff    (c_dim) | q_rope (r_dim) ]
//     with q_eff_h = W_kvb_k_h^T q_nope_h precomputed by a batched GEMV
//   score_h(pos) = q_h . cache(pos) * scale        (MQA: cache shared)
//   ctx_h        = sum_pos softmax(score)_pos * c_kv_norm(pos)   [c_dim]
//     then o_pre_h = W_kvb_v_h ctx_h happens as another batched GEMV.

#include <cuda_fp16.h>

static __device__ __forceinline__ float bf16_to_f32_(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

// Interleaved (GPT-J style) RoPE with a host-precomputed frequency table:
// pairs (2i, 2i+1) of the `half*2`-dim rope slice rotate by pos*inv_freq[i].
// Applied to `n_vecs` vectors of stride `stride`, rope slice at `offset`.
// (DeepSeek's checkpoint stores the rope dims pair-interleaved; both q and
// k use the same transform so scores are consistent.)
extern "C" __global__ void mla_rope_f32(
    float* __restrict__ x,
    int n_vecs,
    int stride,
    int offset,
    int half,
    int pos,
    const float* __restrict__ inv_freq,
    float mscale)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_vecs * half;
         idx += gridDim.x * blockDim.x) {
        const int v = idx / half;
        const int i = idx % half;
        const float ang = (float)pos * inv_freq[i];
        const float c = cosf(ang) * mscale;
        const float s = sinf(ang) * mscale;
        float* p = x + (size_t)v * stride + offset + 2 * i;
        const float x1 = p[0];
        const float x2 = p[1];
        p[0] = x1 * c - x2 * s;
        p[1] = x1 * s + x2 * c;
    }
}

// One block per query head; cache is shared by all heads (MQA). Scores use
// the full qk_dim = c_dim + r_dim row; context accumulates only the first
// c_dim dims. Two-pass softmax with global scratch [heads, max_seq].
extern "C" __global__ void mla_attn_decode_f32(
    const float* __restrict__ q,       // [heads, qk_dim]
    const float* __restrict__ cache,   // [max_seq, qk_dim]
    float* __restrict__ ctx,           // [heads, c_dim]
    float* __restrict__ scratch,       // [heads, max_seq]
    int heads,
    int qk_dim,
    int c_dim,
    int seq_len,
    int max_seq,
    float scale)
{
    const int h = blockIdx.x;
    if (h >= heads) return;
    const float* qh = q + (size_t)h * qk_dim;
    float* sc = scratch + (size_t)h * max_seq;
    __shared__ float red[256];

    float lmax = -1e30f;
    for (int p = threadIdx.x; p < seq_len; p += blockDim.x) {
        const float* cp = cache + (size_t)p * qk_dim;
        float dot = 0.f;
        for (int d = 0; d < qk_dim; ++d) dot += qh[d] * cp[d];
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

    for (int d = threadIdx.x; d < c_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int p = 0; p < seq_len; ++p) {
            acc += sc[p] * cache[(size_t)p * qk_dim + d];
        }
        ctx[(size_t)h * c_dim + d] = acc * invz;
    }
}

// Batched bf16 GEMV: batch b computes y_b = W_b x_b with
// W_b = w + b*w_stride (row-major [rows, cols]), x_b = x + b*x_stride,
// y_b = y + b*y_stride. Strides in elements; x_stride 0 shares the input,
// y_stride lets outputs interleave into a wider per-batch record (e.g. the
// 512-dim absorbed query written into a 576-dim [q_eff | q_rope] row).
// Used per-head for W_kvb_k^T (q absorption) and W_kvb_v (ctx -> v).
extern "C" __global__ void bf16_gemv_batch(
    const unsigned short* __restrict__ w,
    unsigned long long w_stride,
    const float* __restrict__ x,
    int x_stride,
    float* __restrict__ y,
    int y_stride,
    int rows,
    int cols,
    int batch)
{
    const int b = blockIdx.y;
    if (b >= batch) return;
    const unsigned short* wb = w + (size_t)b * w_stride;
    const float* xb = x + (size_t)b * x_stride;
    float* yb = y + (size_t)b * y_stride;

    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const unsigned short* wr = wb + (size_t)row * cols;
    float acc = 0.f;
    for (int c = lane; c < cols; c += 32) {
        acc += bf16_to_f32_(wr[c]) * xb[c];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) yb[row] = acc;
}

// dst[v*dst_stride + dst_off + i] = src[v*src_stride + src_off + i]
// for v in [0, n_vecs), i in [0, n). Strided gather/scatter used to
// assemble MLA query rows and append cache entries.
extern "C" __global__ void strided_copy_f32(
    const float* __restrict__ src,
    int src_stride,
    int src_off,
    float* __restrict__ dst,
    int dst_stride,
    int dst_off,
    int n_vecs,
    int n)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_vecs * n;
         idx += gridDim.x * blockDim.x) {
        const int v = idx / n;
        const int i = idx % n;
        dst[(size_t)v * dst_stride + dst_off + i] =
            src[(size_t)v * src_stride + src_off + i];
    }
}
