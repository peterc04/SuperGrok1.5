/*
 * Grokking Optimizers — ARM NEON SIMD Kernels
 *
 * Hot inner loops accelerated with ARM NEON intrinsics.
 * Targets AArch64 (Apple Silicon, AWS Graviton, etc.)
 */

#include <cmath>
#include <cstring>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
static bool _neon_ok = true;
#else
static bool _neon_ok = false;
#endif

bool simd_available() {
    return _neon_ok;
}


// ═══════════════════════════════════════════════════════════════════
//  NEON Kernels
// ═══════════════════════════════════════════════════════════════════

#if defined(__aarch64__) || defined(_M_ARM64)

static void adam_update_neon(
    float* param, const float* ea, const float* easq,
    float step_size, float bc2, float eps,
    float lr, float wd, int N
) {
    float32x4_t v_step = vdupq_n_f32(-step_size);
    float32x4_t v_bc2 = vdupq_n_f32(bc2);
    float32x4_t v_eps = vdupq_n_f32(eps);
    float32x4_t v_decay = vdupq_n_f32(1.0f - lr * wd);

    int i = 0;
    for (; i + 4 <= N; i += 4) {
        float32x4_t p = vld1q_f32(param + i);
        float32x4_t m = vld1q_f32(ea + i);
        float32x4_t v = vld1q_f32(easq + i);

        p = vmulq_f32(p, v_decay);
        // denom = sqrt(v/bc2) + eps
        float32x4_t denom = vaddq_f32(
            vsqrtq_f32(vdivq_f32(v, v_bc2)),
            v_eps
        );
        // p -= step_size * m / denom
        p = vmlaq_f32(p, v_step, vdivq_f32(m, denom));
        vst1q_f32(param + i, p);
    }
    for (; i < N; i++) {
        param[i] = param[i] * (1.0f - lr * wd)
                   - step_size * ea[i] / (std::sqrt(easq[i] / bc2) + eps);
    }
}

static void lion_update_neon(
    float* param, const float* grad, float* momentum,
    float lr, float beta1, float beta2, float wd, int N
) {
    // NEON doesn't have a direct sign instruction, use comparison
    float32x4_t v_zero = vdupq_n_f32(0.0f);
    float32x4_t v_one = vdupq_n_f32(1.0f);
    float32x4_t v_neg1 = vdupq_n_f32(-1.0f);
    float32x4_t v_lr = vdupq_n_f32(lr);
    float32x4_t v_b1 = vdupq_n_f32(beta1);
    float32x4_t v_1mb1 = vdupq_n_f32(1.0f - beta1);
    float32x4_t v_b2 = vdupq_n_f32(beta2);
    float32x4_t v_1mb2 = vdupq_n_f32(1.0f - beta2);
    float32x4_t v_decay = vdupq_n_f32(1.0f - lr * wd);

    int i = 0;
    for (; i + 4 <= N; i += 4) {
        float32x4_t p = vld1q_f32(param + i);
        float32x4_t g = vld1q_f32(grad + i);
        float32x4_t m = vld1q_f32(momentum + i);

        // interp = beta1*m + (1-beta1)*g
        float32x4_t interp = vmlaq_f32(vmulq_f32(v_b1, m), v_1mb1, g);

        // sign(interp): +1 if > 0, -1 if < 0, 0 if == 0
        uint32x4_t pos = vcgtq_f32(interp, v_zero);
        uint32x4_t neg = vcltq_f32(interp, v_zero);
        float32x4_t sign_v = vbslq_f32(pos, v_one, vbslq_f32(neg, v_neg1, v_zero));

        // p = p * decay - lr * sign
        p = vmulq_f32(p, v_decay);
        p = vmlsq_f32(p, v_lr, sign_v);
        vst1q_f32(param + i, p);

        // m = beta2*m + (1-beta2)*g
        m = vmlaq_f32(vmulq_f32(v_b2, m), v_1mb2, g);
        vst1q_f32(momentum + i, m);
    }
    for (; i < N; i++) {
        float g = grad[i];
        float m_val = momentum[i];
        float interp = beta1 * m_val + (1.0f - beta1) * g;
        float sign_v = (interp > 0.0f) ? 1.0f : ((interp < 0.0f) ? -1.0f : 0.0f);
        param[i] = param[i] * (1.0f - lr * wd) - lr * sign_v;
        momentum[i] = beta2 * m_val + (1.0f - beta2) * g;
    }
}

static void ema_amplify_neon(float* grad, float* ema, float alpha, float lamb, int N) {
    float32x4_t v_alpha = vdupq_n_f32(alpha);
    float32x4_t v_1ma = vdupq_n_f32(1.0f - alpha);
    float32x4_t v_lamb = vdupq_n_f32(lamb);

    int i = 0;
    for (; i + 4 <= N; i += 4) {
        float32x4_t g = vld1q_f32(grad + i);
        float32x4_t e = vld1q_f32(ema + i);

        e = vmlaq_f32(vmulq_f32(v_alpha, e), v_1ma, g);
        vst1q_f32(ema + i, e);

        g = vmlaq_f32(g, v_lamb, e);
        vst1q_f32(grad + i, g);
    }
    for (; i < N; i++) {
        ema[i] = alpha * ema[i] + (1.0f - alpha) * grad[i];
        grad[i] += lamb * ema[i];
    }
}

static void matvec_neon(const float* W, const float* x, float* out,
                        int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        const float* row = W + r * cols;
        int c = 0;
        for (; c + 4 <= cols; c += 4) {
            float32x4_t w = vld1q_f32(row + c);
            float32x4_t v = vld1q_f32(x + c);
            acc = vmlaq_f32(acc, w, v);
        }
        // Horizontal sum
        float sum = vaddvq_f32(acc);
        for (; c < cols; c++)
            sum += row[c] * x[c];
        out[r] = sum;
    }
}

#endif  // __aarch64__


// ═══════════════════════════════════════════════════════════════════
//  Scalar fallback (for non-ARM builds that somehow include this)
// ═══════════════════════════════════════════════════════════════════

static void adam_update_scalar(
    float* param, const float* ea, const float* easq,
    float step_size, float bc2, float eps,
    float lr, float wd, int N
) {
    for (int i = 0; i < N; i++)
        param[i] = param[i] * (1.0f - lr * wd)
                   - step_size * ea[i] / (std::sqrt(easq[i] / bc2) + eps);
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
//  BF16 / INT8 stochastic rounding (NEON + scalar)
//
//  Unbiased quantization for Config 4 CPU optimizer state compression.
// ═══════════════════════════════════════════════════════════════════

static unsigned _sr_philox(unsigned key, unsigned salt) {
    unsigned v = key * 2654435761u + salt * 2246822519u;
    v ^= v >> 16; v *= 0x45d9f3bu; v ^= v >> 16;
    return v;
}

#if defined(__aarch64__) || defined(_M_ARM64)

static void stochastic_round_bf16_neon(
    const float* input, uint16_t* output, int N, unsigned step
) {
    int i = 0;
    for (; i + 4 <= N; i += 4) {
        // Load 4 floats, reinterpret as uint32
        uint32x4_t bits = vreinterpretq_u32_f32(vld1q_f32(input + i));

        // Extract low 16 bits (truncated portion)
        uint32x4_t truncated = vandq_u32(bits, vdupq_n_u32(0xFFFF));

        // Generate random thresholds
        uint32_t rng_vals[4];
        for (int j = 0; j < 4; j++)
            rng_vals[j] = _sr_philox(step, (unsigned)(i + j)) & 0xFFFF;
        uint32x4_t rng = vld1q_u32(rng_vals);

        // Compare: truncated > threshold → round up
        uint32x4_t round_up = vcgtq_u32(truncated, rng);

        // Add 0x10000 where rounding up
        uint32x4_t adj = vandq_u32(round_up, vdupq_n_u32(0x10000));
        bits = vaddq_u32(bits, adj);

        // Shift right 16 to get BF16 bits
        uint16x4_t bf16_bits = vmovn_u32(vshrq_n_u32(bits, 16));
        vst1_u16(output + i, bf16_bits);
    }
    // Scalar tail
    for (; i < N; i++) {
        unsigned bits;
        memcpy(&bits, &input[i], sizeof(float));
        unsigned trunc = bits & 0xFFFF;
        unsigned rng = _sr_philox(step, (unsigned)i) & 0xFFFF;
        if (trunc > rng) bits += 0x10000;
        output[i] = (uint16_t)(bits >> 16);
    }
}

static void stochastic_round_int8_neon(
    const float* input, int8_t* output, const float* scales,
    int N, int block_size, unsigned step
) {
    int i = 0;
    for (; i + 4 <= N; i += 4) {
        float32x4_t vals = vld1q_f32(input + i);

        // Load scales
        float scale_arr[4];
        for (int j = 0; j < 4; j++)
            scale_arr[j] = scales[(i + j) / block_size];
        float32x4_t sc = vld1q_f32(scale_arr);
        sc = vmaxq_f32(sc, vdupq_n_f32(1e-12f));

        // scaled = val / scale
        float32x4_t scaled = vdivq_f32(vals, sc);

        // Process scalar (NEON lacks efficient trunc+stochastic round)
        float s_arr[4];
        vst1q_f32(s_arr, scaled);
        for (int j = 0; j < 4; j++) {
            float sv = s_arr[j];
            float tr = truncf(sv);
            float frac = fabsf(sv - tr);
            float rng = (float)(_sr_philox(step, (unsigned)(i + j)) & 0xFFFF) / 65536.0f;
            if (frac > rng) tr += (sv > 0) ? 1.0f : -1.0f;
            output[i + j] = (int8_t)fmaxf(-127.0f, fminf(127.0f, tr));
        }
    }
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

#endif  // __aarch64__

// Scalar fallback for stochastic rounding
static void stochastic_round_bf16_scalar(
    const float* input, uint16_t* output, int N, unsigned step
) {
    for (int i = 0; i < N; i++) {
        unsigned bits;
        memcpy(&bits, &input[i], sizeof(float));
        unsigned trunc = bits & 0xFFFF;
        unsigned rng = _sr_philox(step, (unsigned)i) & 0xFFFF;
        if (trunc > rng) bits += 0x10000;
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
#if defined(__aarch64__) || defined(_M_ARM64)
    adam_update_neon(param, ea, easq, step_size, bc2, eps, lr, wd, N);
#else
    adam_update_scalar(param, ea, easq, step_size, bc2, eps, lr, wd, N);
#endif
}

void simd_lion_update(float* param, const float* grad, float* momentum,
                      float lr, float beta1, float beta2, float wd, int N) {
#if defined(__aarch64__) || defined(_M_ARM64)
    lion_update_neon(param, grad, momentum, lr, beta1, beta2, wd, N);
#else
    lion_update_scalar(param, grad, momentum, lr, beta1, beta2, wd, N);
#endif
}

void simd_ema_amplify(float* grad, float* ema, float alpha, float lamb, int N) {
#if defined(__aarch64__) || defined(_M_ARM64)
    ema_amplify_neon(grad, ema, alpha, lamb, N);
#else
    ema_amplify_scalar(grad, ema, alpha, lamb, N);
#endif
}

void simd_matvec(const float* W, const float* x, float* out,
                 int rows, int cols) {
#if defined(__aarch64__) || defined(_M_ARM64)
    matvec_neon(W, x, out, rows, cols);
#else
    matvec_scalar(W, x, out, rows, cols);
#endif
}

void simd_stochastic_round_bf16(
    const float* input, uint16_t* output, int N, unsigned step
) {
#if defined(__aarch64__) || defined(_M_ARM64)
    stochastic_round_bf16_neon(input, output, N, step);
#else
    stochastic_round_bf16_scalar(input, output, N, step);
#endif
}

void simd_stochastic_round_int8(
    const float* input, int8_t* output, const float* scales,
    int N, int block_size, unsigned step
) {
#if defined(__aarch64__) || defined(_M_ARM64)
    stochastic_round_int8_neon(input, output, scales, N, block_size, step);
#else
    stochastic_round_int8_scalar(input, output, scales, N, block_size, step);
#endif
}
