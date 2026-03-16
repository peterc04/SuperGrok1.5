/*
 * Grokking Optimizers — AVX-512 SIMD Kernels
 *
 * Hot inner loops accelerated with AVX-512 intrinsics.
 * Falls back to scalar if AVX-512 not available at runtime.
 *
 * Compiled with -march=native; the compiler auto-detects AVX-512 support.
 * Runtime CPUID check gates actual SIMD execution.
 */

#include <cmath>
#include <cstdint>
#include <cstring>

// ═══════════════════════════════════════════════════════════════════
//  CPUID-based runtime feature detection
// ═══════════════════════════════════════════════════════════════════

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>

static bool detect_avx512() {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        return false;
    // AVX-512F is bit 16 of EBX
    return (ebx & (1u << 16)) != 0;
}

static bool _avx512_ok = detect_avx512();
#else
static bool _avx512_ok = false;
#endif

bool simd_available() {
    return _avx512_ok;
}


// ═══════════════════════════════════════════════════════════════════
//  AVX-512 Kernels
// ═══════════════════════════════════════════════════════════════════

#if defined(__AVX512F__)
#include <immintrin.h>

static void adam_update_avx512(
    float* param, const float* ea, const float* easq,
    float step_size, float bc2, float eps,
    float lr, float wd, int N
) {
    __m512 v_step = _mm512_set1_ps(-step_size);
    __m512 v_bc2 = _mm512_set1_ps(bc2);
    __m512 v_eps = _mm512_set1_ps(eps);
    __m512 v_decay = _mm512_set1_ps(1.0f - lr * wd);

    int i = 0;
    for (; i + 16 <= N; i += 16) {
        __m512 p = _mm512_loadu_ps(param + i);
        __m512 m = _mm512_loadu_ps(ea + i);
        __m512 v = _mm512_loadu_ps(easq + i);

        // param = param * decay - step_size * ea / (sqrt(easq/bc2) + eps)
        p = _mm512_mul_ps(p, v_decay);
        __m512 denom = _mm512_add_ps(
            _mm512_sqrt_ps(_mm512_div_ps(v, v_bc2)),
            v_eps
        );
        p = _mm512_fmadd_ps(v_step, _mm512_div_ps(m, denom), p);
        _mm512_storeu_ps(param + i, p);
    }
    // Scalar tail
    for (; i < N; i++) {
        param[i] = param[i] * (1.0f - lr * wd)
                   - step_size * ea[i] / (std::sqrt(easq[i] / bc2) + eps);
    }
}

static void lion_update_avx512(
    float* param, const float* grad, float* momentum,
    float lr, float beta1, float beta2, float wd, int N
) {
    __m512 v_lr = _mm512_set1_ps(lr);
    __m512 v_b1 = _mm512_set1_ps(beta1);
    __m512 v_1mb1 = _mm512_set1_ps(1.0f - beta1);
    __m512 v_b2 = _mm512_set1_ps(beta2);
    __m512 v_1mb2 = _mm512_set1_ps(1.0f - beta2);
    __m512 v_decay = _mm512_set1_ps(1.0f - lr * wd);
    __m512 v_zero = _mm512_setzero_ps();
    __m512 v_one = _mm512_set1_ps(1.0f);
    __m512 v_neg1 = _mm512_set1_ps(-1.0f);

    int i = 0;
    for (; i + 16 <= N; i += 16) {
        __m512 p = _mm512_loadu_ps(param + i);
        __m512 g = _mm512_loadu_ps(grad + i);
        __m512 m = _mm512_loadu_ps(momentum + i);

        // update = sign(beta1*m + (1-beta1)*g)
        __m512 interp = _mm512_fmadd_ps(v_b1, m, _mm512_mul_ps(v_1mb1, g));
        __mmask16 pos_mask = _mm512_cmp_ps_mask(interp, v_zero, _CMP_GT_OQ);
        __mmask16 neg_mask = _mm512_cmp_ps_mask(interp, v_zero, _CMP_LT_OQ);
        __m512 sign_val = _mm512_setzero_ps();
        sign_val = _mm512_mask_mov_ps(sign_val, pos_mask, v_one);
        sign_val = _mm512_mask_mov_ps(sign_val, neg_mask, v_neg1);

        // param = param * decay - lr * sign
        p = _mm512_fmadd_ps(_mm512_sub_ps(v_decay, v_one), p,
                            _mm512_fnmadd_ps(v_lr, sign_val, p));
        // Simplified: p = p * decay - lr * sign
        p = _mm512_loadu_ps(param + i);
        p = _mm512_mul_ps(p, v_decay);
        p = _mm512_fnmadd_ps(v_lr, sign_val, p);
        _mm512_storeu_ps(param + i, p);

        // m = beta2*m + (1-beta2)*g
        m = _mm512_fmadd_ps(v_b2, m, _mm512_mul_ps(v_1mb2, g));
        _mm512_storeu_ps(momentum + i, m);
    }
    // Scalar tail
    for (; i < N; i++) {
        float g = grad[i];
        float m_val = momentum[i];
        float interp = beta1 * m_val + (1.0f - beta1) * g;
        float sign_v = (interp > 0.0f) ? 1.0f : ((interp < 0.0f) ? -1.0f : 0.0f);
        param[i] = param[i] * (1.0f - lr * wd) - lr * sign_v;
        momentum[i] = beta2 * m_val + (1.0f - beta2) * g;
    }
}

static void ema_amplify_avx512(float* grad, float* ema, float alpha, float lamb, int N) {
    __m512 v_alpha = _mm512_set1_ps(alpha);
    __m512 v_1ma = _mm512_set1_ps(1.0f - alpha);
    __m512 v_lamb = _mm512_set1_ps(lamb);

    int i = 0;
    for (; i + 16 <= N; i += 16) {
        __m512 g = _mm512_loadu_ps(grad + i);
        __m512 e = _mm512_loadu_ps(ema + i);

        // ema = alpha*ema + (1-alpha)*grad
        e = _mm512_fmadd_ps(v_alpha, e, _mm512_mul_ps(v_1ma, g));
        _mm512_storeu_ps(ema + i, e);

        // grad += lamb * ema
        g = _mm512_fmadd_ps(v_lamb, e, g);
        _mm512_storeu_ps(grad + i, g);
    }
    for (; i < N; i++) {
        ema[i] = alpha * ema[i] + (1.0f - alpha) * grad[i];
        grad[i] += lamb * ema[i];
    }
}

static void matvec_avx512(const float* W, const float* x, float* out,
                          int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        __m512 acc = _mm512_setzero_ps();
        const float* row = W + r * cols;
        int c = 0;
        for (; c + 16 <= cols; c += 16) {
            __m512 w = _mm512_loadu_ps(row + c);
            __m512 v = _mm512_loadu_ps(x + c);
            acc = _mm512_fmadd_ps(w, v, acc);
        }
        float sum = _mm512_reduce_add_ps(acc);
        for (; c < cols; c++)
            sum += row[c] * x[c];
        out[r] = sum;
    }
}

#endif  // __AVX512F__


// ═══════════════════════════════════════════════════════════════════
//  Scalar fallback implementations
// ═══════════════════════════════════════════════════════════════════

static void adam_update_scalar(
    float* param, const float* ea, const float* easq,
    float step_size, float bc2, float eps,
    float lr, float wd, int N
) {
    for (int i = 0; i < N; i++) {
        param[i] = param[i] * (1.0f - lr * wd)
                   - step_size * ea[i] / (std::sqrt(easq[i] / bc2) + eps);
    }
}

static void lion_update_scalar(
    float* param, const float* grad, float* momentum,
    float lr, float beta1, float beta2, float wd, int N
) {
    for (int i = 0; i < N; i++) {
        float g = grad[i];
        float m = momentum[i];
        float interp = beta1 * m + (1.0f - beta1) * g;
        float sign_v = (interp > 0.0f) ? 1.0f : ((interp < 0.0f) ? -1.0f : 0.0f);
        param[i] = param[i] * (1.0f - lr * wd) - lr * sign_v;
        momentum[i] = beta2 * m + (1.0f - beta2) * g;
    }
}

static void ema_amplify_scalar(float* grad, float* ema, float alpha, float lamb, int N) {
    for (int i = 0; i < N; i++) {
        ema[i] = alpha * ema[i] + (1.0f - alpha) * grad[i];
        grad[i] += lamb * ema[i];
    }
}

// ═══════════════════════════════════════════════════════════════════
//  BF16 stochastic rounding (AVX-512)
//
//  Unbiased quantization: add random fraction of ULP before truncation.
//  Used by Config 4 CPU kernels for optimizer state compression.
// ═══════════════════════════════════════════════════════════════════

#if defined(__AVX512F__) && defined(__AVX512BW__)

static unsigned _sr_philox(unsigned key, unsigned salt) {
    unsigned v = key * 2654435761u + salt * 2246822519u;
    v ^= v >> 16; v *= 0x45d9f3bu; v ^= v >> 16;
    return v;
}

static void stochastic_round_bf16_avx512(
    const float* input, uint16_t* output, int N, unsigned step
) {
    // Process 16 floats at a time
    int i = 0;
    for (; i + 16 <= N; i += 16) {
        __m512 vals = _mm512_loadu_ps(input + i);
        __m512i bits = _mm512_castps_si512(vals);

        // Extract truncated (lower 16) bits: these are what BF16 drops
        __m512i truncated = _mm512_and_si512(bits, _mm512_set1_epi32(0xFFFF));

        // Generate 16 random thresholds using Philox
        __m512i rng;
        {
            unsigned rng_vals[16];
            for (int j = 0; j < 16; j++)
                rng_vals[j] = _sr_philox(step, (unsigned)(i + j)) & 0xFFFF;
            rng = _mm512_loadu_si512(rng_vals);
        }

        // Compare: if truncated > threshold, round up
        __mmask16 round_up = _mm512_cmpgt_epu32_mask(truncated, rng);

        // Add 0x10000 to round up, then shift right 16 to get BF16
        __m512i rounded = _mm512_mask_add_epi32(bits, round_up, bits,
                                                 _mm512_set1_epi32(0x10000));
        __m512i bf16_bits = _mm512_srli_epi32(rounded, 16);

        // Pack 16 x 32-bit → 16 x 16-bit
        // Extract low 16 bits of each 32-bit element
        uint32_t tmp[16];
        _mm512_storeu_si512(tmp, bf16_bits);
        for (int j = 0; j < 16; j++)
            output[i + j] = (uint16_t)tmp[j];
    }
    // Scalar tail
    for (; i < N; i++) {
        unsigned bits;
        memcpy(&bits, &input[i], sizeof(float));
        unsigned trunc = bits & 0xFFFF;
        unsigned rng = _sr_philox(step, (unsigned)i) & 0xFFFF;
        if (trunc > rng)
            bits += 0x10000;
        output[i] = (uint16_t)(bits >> 16);
    }
}

static void stochastic_round_int8_avx512(
    const float* input, int8_t* output, const float* scales,
    int N, int block_size, unsigned step
) {
    int i = 0;
    for (; i + 16 <= N; i += 16) {
        __m512 vals = _mm512_loadu_ps(input + i);

        // Load per-element block scales
        float scale_arr[16];
        for (int j = 0; j < 16; j++)
            scale_arr[j] = scales[(i + j) / block_size];
        __m512 sc = _mm512_loadu_ps(scale_arr);

        // scaled = val / scale
        __m512 scaled = _mm512_div_ps(vals, _mm512_max_ps(sc,
                                       _mm512_set1_ps(1e-12f)));
        __m512 trunc_val = _mm512_roundscale_ps(scaled,
                           _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m512 frac = _mm512_abs_ps(_mm512_sub_ps(scaled, trunc_val));

        // Generate random thresholds
        float rng_vals[16];
        for (int j = 0; j < 16; j++)
            rng_vals[j] = (float)(_sr_philox(step, (unsigned)(i + j)) & 0xFFFF) / 65536.0f;
        __m512 threshold = _mm512_loadu_ps(rng_vals);

        // Stochastic round: if frac > threshold, round away from zero
        __mmask16 round_up = _mm512_cmp_ps_mask(frac, threshold, _CMP_GT_OQ);
        __m512 sign = _mm512_and_ps(scaled, _mm512_set1_ps(-0.0f));
        __m512 adj = _mm512_mask_blend_ps(round_up,
                                           _mm512_setzero_ps(),
                                           _mm512_or_ps(_mm512_set1_ps(1.0f), sign));
        __m512 result = _mm512_add_ps(trunc_val, adj);

        // Clamp to [-127, 127]
        result = _mm512_max_ps(result, _mm512_set1_ps(-127.0f));
        result = _mm512_min_ps(result, _mm512_set1_ps(127.0f));

        // Convert to int8
        __m512i ints = _mm512_cvtps_epi32(result);
        // Pack 32-bit → 8-bit
        int32_t tmp[16];
        _mm512_storeu_si512(tmp, ints);
        for (int j = 0; j < 16; j++)
            output[i + j] = (int8_t)tmp[j];
    }
    // Scalar tail
    for (; i < N; i++) {
        float scale = scales[i / block_size];
        float scaled = input[i] / fmaxf(scale, 1e-12f);
        float tr = truncf(scaled);
        float frac = fabsf(scaled - tr);
        float rng = (float)(_sr_philox(step, (unsigned)i) & 0xFFFF) / 65536.0f;
        if (frac > rng) tr += (scaled > 0) ? 1.0f : -1.0f;
        output[i] = (int8_t)fmaxf(-127.0f, fminf(127.0f, tr));
    }
}

#endif  // __AVX512F__ && __AVX512BW__

// Scalar fallback implementations for stochastic rounding
static void stochastic_round_bf16_scalar(
    const float* input, uint16_t* output, int N, unsigned step
) {
    for (int i = 0; i < N; i++) {
        unsigned bits;
        memcpy(&bits, &input[i], sizeof(float));
        unsigned trunc = bits & 0xFFFF;
        unsigned rng = _sr_philox(step, (unsigned)i) & 0xFFFF;
        if (trunc > rng)
            bits += 0x10000;
        output[i] = (uint16_t)(bits >> 16);
    }
}

static void stochastic_round_int8_scalar(
    const float* input, int8_t* output, const float* scales,
    int N, int block_size, unsigned step
) {
    for (int i = 0; i < N; i++) {
        float scale = scales[i / block_size];
        float scaled = input[i] / fmaxf(scale, 1e-12f);
        float tr = truncf(scaled);
        float frac = fabsf(scaled - tr);
        float rng = (float)(_sr_philox(step, (unsigned)i) & 0xFFFF) / 65536.0f;
        if (frac > rng) tr += (scaled > 0) ? 1.0f : -1.0f;
        output[i] = (int8_t)fmaxf(-127.0f, fminf(127.0f, tr));
    }
}

static void matvec_scalar(const float* W, const float* x, float* out,
                          int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++)
            sum += W[r * cols + c] * x[c];
        out[r] = sum;
    }
}


// ═══════════════════════════════════════════════════════════════════
//  Public dispatch functions
// ═══════════════════════════════════════════════════════════════════

void simd_adam_update(float* param, const float* ea, const float* easq,
                      float step_size, float bc2, float eps,
                      float lr, float wd, int N) {
#if defined(__AVX512F__)
    if (_avx512_ok) {
        adam_update_avx512(param, ea, easq, step_size, bc2, eps, lr, wd, N);
        return;
    }
#endif
    adam_update_scalar(param, ea, easq, step_size, bc2, eps, lr, wd, N);
}

void simd_lion_update(float* param, const float* grad, float* momentum,
                      float lr, float beta1, float beta2, float wd, int N) {
#if defined(__AVX512F__)
    if (_avx512_ok) {
        lion_update_avx512(param, grad, momentum, lr, beta1, beta2, wd, N);
        return;
    }
#endif
    lion_update_scalar(param, grad, momentum, lr, beta1, beta2, wd, N);
}

void simd_ema_amplify(float* grad, float* ema, float alpha, float lamb, int N) {
#if defined(__AVX512F__)
    if (_avx512_ok) {
        ema_amplify_avx512(grad, ema, alpha, lamb, N);
        return;
    }
#endif
    ema_amplify_scalar(grad, ema, alpha, lamb, N);
}

void simd_matvec(const float* W, const float* x, float* out,
                 int rows, int cols) {
#if defined(__AVX512F__)
    if (_avx512_ok) {
        matvec_avx512(W, x, out, rows, cols);
        return;
    }
#endif
    matvec_scalar(W, x, out, rows, cols);
}

void simd_stochastic_round_bf16(
    const float* input, uint16_t* output, int N, unsigned step
) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (_avx512_ok) {
        stochastic_round_bf16_avx512(input, output, N, step);
        return;
    }
#endif
    stochastic_round_bf16_scalar(input, output, N, step);
}

void simd_stochastic_round_int8(
    const float* input, int8_t* output, const float* scales,
    int N, int block_size, unsigned step
) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (_avx512_ok) {
        stochastic_round_int8_avx512(input, output, scales, N, block_size, step);
        return;
    }
#endif
    stochastic_round_int8_scalar(input, output, scales, N, block_size, step);
}
