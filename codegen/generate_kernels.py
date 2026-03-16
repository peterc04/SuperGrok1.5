#!/usr/bin/env python3
"""
SuperGrok v2 — Kernel Code Generator

Reads optimizer specifications from kernel_specs.yaml and generates CUDA
kernel source files for every template-able variant (Config 4, MoE, vec4).

Usage:
    python codegen/generate_kernels.py --output csrc/cuda/generated/

Generated files go to csrc/cuda/generated/ and are compiled by setup.py.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

# ═══════════════════════════════════════════════════════════════════════
#  Variant axis parsing
# ═══════════════════════════════════════════════════════════════════════

def parse_variant(variant_str):
    """Parse variant string like 'S_F_D' into axis dict.

    GPU format: Vec_State_Model  (e.g. S_F_D, V_Q_M)
    CPU format: cpu_State        (e.g. cpu_F, cpu_Q)
    """
    parts = variant_str.split('_')

    if parts[0] == 'cpu':
        return {
            'platform': 'cpu',
            'vec4': False,
            'config4': parts[1] == 'Q',
            'moe': False,
        }

    return {
        'platform': 'gpu',
        'vec4': parts[0] == 'V',
        'config4': parts[1] == 'Q',
        'moe': parts[2] == 'M',
    }


def kernel_suffix(v):
    """Generate kernel name suffix from variant dict."""
    parts = []
    if v['platform'] == 'cpu':
        parts.append('cpu')
        if v['config4']:
            parts.append('q4')
        return '_'.join(parts)

    if v['moe']:
        parts.append('moe')
    if v['config4']:
        parts.append('q4')
    if v['vec4']:
        parts.append('vec4')
    return '_'.join(parts) if parts else ''


# ═══════════════════════════════════════════════════════════════════════
#  Config 4 state parameter generation
# ═══════════════════════════════════════════════════════════════════════

def q4_params_for_state(state_name, q4_format):
    """Generate kernel parameter declarations for Config 4 state."""
    if q4_format == 'int8_block8':
        return [
            f'int8_t* __restrict__ {state_name}_q',
            f'float* __restrict__ {state_name}_scales',
        ]
    elif q4_format == 'int4_block32':
        return [
            f'int8_t* __restrict__ {state_name}_q',  # packed nibbles
            f'float* __restrict__ {state_name}_scales',
        ]
    elif q4_format == 'bf16_sr':
        return [
            f'__nv_bfloat16* __restrict__ {state_name}_bf16',
        ]
    return []


def fp32_params_for_state(state_name):
    """Generate kernel parameter declarations for FP32 state."""
    return [f'float* __restrict__ {state_name}']


# ═══════════════════════════════════════════════════════════════════════
#  Config 4 load/store code generation (inline C++)
# ═══════════════════════════════════════════════════════════════════════

def gen_q4_load(state_name, q4_format, idx, component=''):
    """Generate C++ code for loading a Config 4 quantized state value."""
    sfx = f'_{component}' if component else ''
    if q4_format == 'int8_block8':
        return (
            f'float {state_name}_val{sfx};\n'
            f'    {{\n'
            f'        const int _blk = {idx} / 8;\n'
            f'        {state_name}_val{sfx} = (float){state_name}_q[{idx}] * {state_name}_scales[_blk];\n'
            f'    }}'
        )
    elif q4_format == 'int4_block32':
        return (
            f'float {state_name}_val{sfx};\n'
            f'    {{\n'
            f'        const int _blk = {idx} / 32;\n'
            f'        const int _byte_idx = {idx} / 2;\n'
            f'        const int _nibble = {idx} % 2;\n'
            f'        int8_t _packed = {state_name}_q[_byte_idx];\n'
            f'        int8_t _raw = _nibble ? ((_packed >> 4) & 0x0F) : (_packed & 0x0F);\n'
            f'        if (_raw & 0x08) _raw |= (int8_t)0xF0;\n'
            f'        {state_name}_val{sfx} = (float)_raw * {state_name}_scales[_blk];\n'
            f'    }}'
        )
    elif q4_format == 'bf16_sr':
        return f'float {state_name}_val{sfx} = __bfloat162float({state_name}_bf16[{idx}]);'
    return ''


def gen_q4_store(state_name, q4_format, idx, val, rng_counter, component=''):
    """Generate C++ code for storing a Config 4 quantized state value."""
    if q4_format == 'int8_block8':
        return (
            f'    {{\n'
            f'        const int _blk = {idx} / 8;\n'
            f'        float _new_scale = fmaxf(fabsf({val}), 1e-12f) / 127.0f;\n'
            f'        unsigned _rng = philox_hash(global_step * {rng_counter}u, (unsigned){idx});\n'
            f'        {state_name}_q[{idx}] = float_to_int8_stochastic({val}, _new_scale, _rng);\n'
            f'        if ({idx} % 8 == 0) {state_name}_scales[_blk] = _new_scale;\n'
            f'    }}'
        )
    elif q4_format == 'int4_block32':
        return (
            f'    {{\n'
            f'        const int _blk = {idx} / 32;\n'
            f'        float _new_scale = fmaxf(fabsf({val}), 1e-12f) / 7.0f;\n'
            f'        unsigned _rng = philox_hash(global_step * {rng_counter}u, (unsigned){idx});\n'
            f'        int8_t _nib = float_to_int4_stochastic({val}, _new_scale, _rng);\n'
            f'        const int _byte_idx = {idx} / 2;\n'
            f'        const int _which = {idx} % 2;\n'
            f'        int8_t _old = {state_name}_q[_byte_idx];\n'
            f'        {state_name}_q[_byte_idx] = _which ?\n'
            f'            ((_old & 0x0F) | ((_nib & 0x0F) << 4)) :\n'
            f'            ((_old & 0xF0) | (_nib & 0x0F));\n'
            f'        if ({idx} % 32 == 0) {state_name}_scales[_blk] = _new_scale;\n'
            f'    }}'
        )
    elif q4_format == 'bf16_sr':
        return (
            f'    {{\n'
            f'        unsigned _rng = philox_hash(global_step * {rng_counter}u + 1u, (unsigned){idx});\n'
            f'        {state_name}_bf16[{idx}] = float_to_bf16_stochastic({val}, _rng);\n'
            f'    }}'
        )
    return ''


def gen_fp32_load(state_name, idx, component=''):
    """Generate C++ code for non-temporal FP32 state load."""
    sfx = f'_{component}' if component else ''
    return f'const float {state_name}_val{sfx} = stream_load(&{state_name}[{idx}]);'


def gen_fp32_store(state_name, idx, val, component=''):
    """Generate C++ code for non-temporal FP32 state store."""
    return f'stream_store(&{state_name}[{idx}], {val});'


# ═══════════════════════════════════════════════════════════════════════
#  Element-wise kernel generator
# ═══════════════════════════════════════════════════════════════════════

def generate_elementwise_kernel(optimizer_name, spec, variant):
    """Generate a complete .cu kernel for an element-wise optimizer variant."""
    v = parse_variant(variant)
    suffix = kernel_suffix(v)
    kern_name = f'{optimizer_name}_step'
    if suffix:
        kern_name += f'_{suffix}'

    is_vec4 = v['vec4']
    is_q4 = v['config4']
    is_moe = v['moe']
    is_cpu = v['platform'] == 'cpu'

    states = spec.get('states', {})
    scalars = spec.get('scalars', [])
    has_adam = spec.get('has_adam', False)
    block_size = spec.get('block_size', 256)
    lb_threads, lb_blocks = spec.get('launch_bounds', [256, 8])
    extra_inputs = spec.get('extra_inputs', {})
    grad_name = spec.get('grad_name', 'grad')

    lines = []

    # ─── File header ────────────────────────────────────────────────
    lines.append(f'/* GENERATED by codegen/generate_kernels.py — DO NOT EDIT */')
    lines.append(f'/* Optimizer: {optimizer_name}, Variant: {variant} */')
    lines.append('')

    if is_cpu:
        return _generate_cpu_kernel(optimizer_name, spec, variant, kern_name, v)

    lines.append('#include <torch/extension.h>')
    lines.append('#include "platform.h"')
    lines.append('#include "utils.cuh"')
    lines.append('')

    # ─── Philox RNG + SR helpers (only for Q4) ─────────────────────
    if is_q4:
        lines.append('// Philox RNG (stateless)')
        lines.append('__device__ __forceinline__ unsigned philox_hash(unsigned key, unsigned salt) {')
        lines.append('    unsigned v = key * 2654435761u + salt * 2246822519u;')
        lines.append('    v ^= v >> 16; v *= 0x45d9f3bu; v ^= v >> 16;')
        lines.append('    return v;')
        lines.append('}')
        lines.append('')
        lines.append('// INT4 stochastic rounding')
        lines.append('__device__ __forceinline__ int8_t float_to_int4_stochastic(')
        lines.append('    float val, float scale, unsigned rand_bits) {')
        lines.append('    float scaled = val / fmaxf(scale, 1e-12f);')
        lines.append('    float tr = truncf(scaled);')
        lines.append('    float frac = fabsf(scaled - tr);')
        lines.append('    float threshold = (float)(rand_bits & 0xFFFF) / 65536.0f;')
        lines.append('    if (frac > threshold) tr += (scaled > 0) ? 1.0f : -1.0f;')
        lines.append('    return (int8_t)fmaxf(-7.0f, fminf(7.0f, tr));')
        lines.append('}')
        lines.append('')

    # ─── Kernel signature ──────────────────────────────────────────
    if is_vec4:
        lines.append(f'__launch_bounds__({lb_threads}, {lb_blocks})')
        lines.append(f'__global__ void {kern_name}_kernel(')
        # Param + grad as float4
        lines.append(f'    float4* __restrict__ param4,')
        lines.append(f'    const float4* __restrict__ {grad_name}4,')
        # States
        for sname, sconf in states.items():
            if is_q4:
                for p in q4_params_for_state(sname, sconf['q4_format']):
                    lines.append(f'    {p},')
            else:
                lines.append(f'    float4* __restrict__ {sname}4,')
        # Extra inputs
        for ename, econf in extra_inputs.items():
            lines.append(f'    const float4* __restrict__ {ename}4,')
        # MoE mask
        if is_moe:
            lines.append(f'    const bool* __restrict__ active_mask,')
        # Scalars
        for sc in scalars:
            lines.append(f'    const float {sc},')
        if is_q4:
            lines.append(f'    const unsigned global_step,')
        lines.append(f'    const int N4')
        lines.append(') {')
        lines.append(f'    const int i = blockIdx.x * blockDim.x + threadIdx.x;')
        lines.append(f'    if (i >= N4) return;')
        lines.append('')
        if is_moe:
            lines.append(f'    // MoE early-exit')
            lines.append(f'    if (!active_mask[i*4] && !active_mask[i*4+1] &&')
            lines.append(f'        !active_mask[i*4+2] && !active_mask[i*4+3]) return;')
            lines.append('')

        # Generate vec4 body by processing 4 components
        _gen_vec4_body(lines, optimizer_name, spec, v)
        lines.append('}')
    else:
        # Scalar kernel
        lines.append(f'template <typename scalar_t>')
        lines.append(f'__launch_bounds__({lb_threads}, {lb_blocks})')
        lines.append(f'__global__ void {kern_name}_kernel(')
        lines.append(f'    scalar_t* __restrict__ param,')
        lines.append(f'    const scalar_t* __restrict__ {grad_name},')
        for sname, sconf in states.items():
            if is_q4:
                for p in q4_params_for_state(sname, sconf['q4_format']):
                    lines.append(f'    {p},')
            else:
                lines.append(f'    float* __restrict__ {sname},')
        for ename, econf in extra_inputs.items():
            lines.append(f'    {econf["type"]} __restrict__ {ename},')
        if is_moe:
            lines.append(f'    const bool* __restrict__ active_mask,')
        for sc in scalars:
            lines.append(f'    const float {sc},')
        if is_q4:
            lines.append(f'    const unsigned global_step,')
        lines.append(f'    const int N')
        lines.append(') {')
        lines.append(f'    const int idx = blockIdx.x * blockDim.x + threadIdx.x;')
        lines.append(f'    if (idx >= N) return;')
        lines.append('')
        if is_moe:
            lines.append(f'    // MoE early-exit: skip inactive expert parameters')
            lines.append(f'    if (!active_mask[idx]) return;')
            lines.append('')

        # Generate scalar body
        _gen_scalar_body(lines, optimizer_name, spec, v)
        lines.append('}')

    lines.append('')

    # ─── Launcher ──────────────────────────────────────────────────
    _gen_launcher(lines, optimizer_name, kern_name, spec, v)

    return '\n'.join(lines) + '\n'


def _gen_scalar_body(lines, optimizer_name, spec, v):
    """Generate the scalar kernel body."""
    states = spec.get('states', {})
    has_adam = spec.get('has_adam', False)
    is_q4 = v['config4']
    grad_name = spec.get('grad_name', 'grad')
    extra_inputs = spec.get('extra_inputs', {})

    # Read gradient
    lines.append(f'    const float g = static_cast<float>({grad_name}[idx]);')
    lines.append('')

    if has_adam:
        effective_grad = spec.get('effective_grad', 'g')
        pre_adam = spec.get('pre_adam_math', '')

        # Pre-Adam math (EMA filter, amplification, etc.)
        if pre_adam:
            for line in pre_adam.strip().split('\n'):
                l = line.strip()
                if not l:
                    lines.append('')
                    continue
                # Replace STATE_LOAD / STATE_STORE macros
                l = _replace_state_macros(l, states, is_q4, 'idx')
                # Replace EXTRA_LOAD macros
                for ename in extra_inputs:
                    l = l.replace(f'EXTRA_LOAD({ename}, idx)',
                                  f'static_cast<float>({ename}[idx])')
                lines.append(f'    {l}')
            lines.append('')

        effective_grad_sq = spec.get('effective_grad_sq', f'{effective_grad} * {effective_grad}')
        adam_denom_extra = spec.get('adam_denom_extra', '')
        adam_decay_extra = spec.get('adam_decay_extra', '')

        # Adam moment loads
        for sname in ['exp_avg', 'exp_avg_sq']:
            if sname in states:
                if is_q4:
                    lines.append(f'    {gen_q4_load(sname, states[sname]["q4_format"], "idx")}')
                else:
                    lines.append(f'    {gen_fp32_load(sname, "idx")}')
        lines.append('')

        # Adam moment updates
        lines.append(f'    const float ea = beta1 * exp_avg_val + (1.0f - beta1) * {effective_grad};')
        lines.append(f'    const float easq = beta2 * exp_avg_sq_val + (1.0f - beta2) * {effective_grad_sq};')
        lines.append('')

        # Adam moment stores
        rng_ctr = 2
        for sname in ['exp_avg', 'exp_avg_sq']:
            if sname in states:
                val = 'ea' if sname == 'exp_avg' else 'easq'
                if is_q4:
                    lines.append(f'    {gen_q4_store(sname, states[sname]["q4_format"], "idx", val, rng_ctr)}')
                    rng_ctr += 1
                else:
                    lines.append(f'    {gen_fp32_store(sname, "idx", val)}')
        lines.append('')

        # Adam step with weight decay
        lines.append(f'    const float step_size = lr / bc1;')
        lines.append(f'    const float denom = sqrtf(easq / bc2) + {adam_denom_extra}eps;')
        lines.append(f'    float p = static_cast<float>(param[idx]);')
        lines.append(f'    p *= (1.0f - lr * {adam_decay_extra}weight_decay);')
        lines.append(f'    p -= step_size * ea / denom;')
        lines.append(f'    param[idx] = static_cast<scalar_t>(p);')

    else:
        # Custom update math
        update_math = spec.get('update_math', '')
        if update_math:
            for line in update_math.strip().split('\n'):
                l = line.strip()
                if not l:
                    lines.append('')
                    continue
                l = _replace_state_macros(l, states, is_q4, 'idx')
                l = _replace_param_macros(l)
                l = _replace_grad_macros(l, grad_name)
                for ename in extra_inputs:
                    l = l.replace(f'EXTRA_LOAD({ename}, idx)',
                                  f'static_cast<float>({ename}[idx])')
                lines.append(f'    {l}')


def _gen_vec4_body(lines, optimizer_name, spec, v):
    """Generate the vec4 kernel body (processes 4 elements per thread)."""
    states = spec.get('states', {})
    has_adam = spec.get('has_adam', False)
    is_q4 = v['config4']
    grad_name = spec.get('grad_name', 'grad')
    extra_inputs = spec.get('extra_inputs', {})

    # Read param and grad as float4
    lines.append(f'    float4 p = param4[i];')
    lines.append(f'    float4 g = {grad_name}4[i];')

    # Read extra inputs
    for ename in extra_inputs:
        lines.append(f'    float4 {ename}_v = {ename}4[i];')

    # Read states
    for sname, sconf in states.items():
        if is_q4:
            # Q4 vec4: process 4 consecutive elements using scalar Q4 ops
            for c_idx, c_name in enumerate(['x', 'y', 'z', 'w']):
                lines.append(f'    {gen_q4_load(sname, sconf["q4_format"], f"i*4+{c_idx}", c_name)}')
        else:
            lines.append(f'    float4 {sname}_v = stream_load4(&{sname}4[i]);')
    lines.append('')

    if has_adam:
        effective_grad = spec.get('effective_grad', 'g')
        pre_adam = spec.get('pre_adam_math', '')
        effective_grad_sq = spec.get('effective_grad_sq', '')
        adam_denom_extra = spec.get('adam_denom_extra', '')
        adam_decay_extra = spec.get('adam_decay_extra', '')

        # Pre-Adam math for each component
        if pre_adam:
            for c_idx, c_name in enumerate(['x', 'y', 'z', 'w']):
                lines.append(f'    // Component {c_name}')
                for line in pre_adam.strip().split('\n'):
                    l = line.strip()
                    if not l or l.startswith('//'):
                        continue
                    l = _vec4_replace(l, states, is_q4, f'i*4+{c_idx}', c_name, extra_inputs)
                    lines.append(f'    {l}')
            lines.append('')

        # Adam update for each component
        for c_idx, c_name in enumerate(['x', 'y', 'z', 'w']):
            idx_str = f'i*4+{c_idx}'
            if is_q4:
                ea_val = f'exp_avg_val_{c_name}'
                easq_val = f'exp_avg_sq_val_{c_name}'
            else:
                ea_val = f'exp_avg_v.{c_name}'
                easq_val = f'exp_avg_sq_v.{c_name}'

            eff = _vec4_effective(effective_grad, c_name, extra_inputs)
            eff_sq = _vec4_effective(effective_grad_sq, c_name, extra_inputs) if effective_grad_sq else f'{eff} * {eff}'

            lines.append(f'    const float ea_{c_name} = beta1 * {ea_val} + (1.0f - beta1) * {eff};')
            lines.append(f'    const float easq_{c_name} = beta2 * {easq_val} + (1.0f - beta2) * {eff_sq};')

        lines.append('')

        # Store moments
        rng_ctr = 2
        if is_q4:
            for sname in ['exp_avg', 'exp_avg_sq']:
                if sname in states:
                    for c_idx, c_name in enumerate(['x', 'y', 'z', 'w']):
                        val = f'ea_{c_name}' if sname == 'exp_avg' else f'easq_{c_name}'
                        lines.append(f'    {gen_q4_store(sname, states[sname]["q4_format"], f"i*4+{c_idx}", val, rng_ctr)}')
                    rng_ctr += 1
        else:
            for sname in ['exp_avg', 'exp_avg_sq']:
                if sname in states:
                    var = 'ea' if sname == 'exp_avg' else 'easq'
                    lines.append(f'    float4 {sname}_out;')
                    for c_name in ['x', 'y', 'z', 'w']:
                        lines.append(f'    {sname}_out.{c_name} = {var}_{c_name};')
                    lines.append(f'    stream_store4(&{sname}4[i], {sname}_out);')
        lines.append('')

        # Adam step for each component
        lines.append(f'    const float step_size = lr / bc1;')
        lines.append(f'    const float decay = 1.0f - lr * {adam_decay_extra}weight_decay;')
        for c_name in ['x', 'y', 'z', 'w']:
            lines.append(f'    p.{c_name} = decay * p.{c_name} - step_size * ea_{c_name} / (sqrtf(easq_{c_name} / bc2) + {adam_denom_extra}eps);')
        lines.append(f'    param4[i] = p;')

    else:
        # Non-Adam: generate vec4 custom update
        update_math = spec.get('update_math', '')
        if update_math:
            for c_idx, c_name in enumerate(['x', 'y', 'z', 'w']):
                lines.append(f'    // Component {c_name}')
                lines.append(f'    {{')
                idx_str = f'i*4+{c_idx}'
                for line in update_math.strip().split('\n'):
                    l = line.strip()
                    if not l:
                        continue
                    l = _vec4_replace(l, states, is_q4, idx_str, c_name, extra_inputs)
                    # Replace PARAM_LOAD/PARAM_STORE with component-scoped names
                    l = l.replace('PARAM_LOAD(idx)', f'p.{c_name}')
                    import re
                    m = re.search(r'PARAM_STORE\(idx,\s*(.+?)\)', l)
                    if m:
                        val = m.group(1)
                        l = l.replace(m.group(0), f'p.{c_name} = {val}')
                    m = re.search(r'GRAD_STORE\(idx,\s*(.+?)\)', l)
                    if m:
                        val = m.group(1)
                        gn = spec.get('grad_name', 'grad')
                        l = l.replace(m.group(0), f'{gn}4[i].{c_name} = {val}')
                    # Rename local 'float p =' to 'float p_c =' to avoid shadowing float4 p
                    l = l.replace('float p = ', f'float p_{c_name} = ')
                    l = l.replace('float p;', f'float p_{c_name};')
                    # Fix references to local p that should be p_{c_name}
                    # Only replace standalone 'p' in expressions (not p. or p4)
                    if f'float p_{c_name}' in l or f'p_{c_name}' in l:
                        pass  # already renamed
                    elif 'p = p -' in l or 'p -= ' in l or 'p *= ' in l:
                        l = l.replace('p = p -', f'p_{c_name} = p_{c_name} -')
                        l = l.replace('p -= ', f'p_{c_name} -= ')
                        l = l.replace('p *= ', f'p_{c_name} *= ')
                    lines.append(f'        {l}')
                lines.append(f'    }}')


def _replace_state_macros(line, states, is_q4, idx):
    """Replace STATE_LOAD/STATE_STORE macros in update math."""
    for sname, sconf in states.items():
        # STATE_LOAD(name, idx) → appropriate load
        load_pat = f'STATE_LOAD({sname}, {idx})'
        if load_pat in line:
            if is_q4:
                line = line.replace(load_pat, f'{sname}_val')
            else:
                line = line.replace(load_pat, f'stream_load(&{sname}[{idx}])')

        # STATE_STORE(name, idx, val)
        import re
        store_pat = f'STATE_STORE({sname}, {idx}, '
        if store_pat in line:
            # Extract value expression
            start = line.index(store_pat) + len(store_pat)
            depth = 1
            end = start
            while end < len(line) and depth > 0:
                if line[end] == '(':
                    depth += 1
                elif line[end] == ')':
                    depth -= 1
                end += 1
            val_expr = line[start:end-1]
            full_macro = line[line.index(f'STATE_STORE({sname}'):end]
            if is_q4:
                replacement = f'/* q4 store {sname} */ {sname}_val = {val_expr}; /* stored below */'
            else:
                replacement = f'stream_store(&{sname}[{idx}], {val_expr})'
            line = line.replace(full_macro, replacement)
    return line


def _replace_param_macros(line):
    """Replace PARAM_LOAD/PARAM_STORE macros."""
    line = line.replace('PARAM_LOAD(idx)', 'static_cast<float>(param[idx])')
    import re
    m = re.search(r'PARAM_STORE\(idx,\s*(.+?)\)', line)
    if m:
        val = m.group(1)
        line = line.replace(m.group(0), f'param[idx] = static_cast<scalar_t>({val})')
    return line


def _replace_grad_macros(line, grad_name):
    """Replace GRAD_STORE macros."""
    import re
    m = re.search(r'GRAD_STORE\(idx,\s*(.+?)\)', line)
    if m:
        val = m.group(1)
        line = line.replace(m.group(0), f'{grad_name}[idx] = static_cast<scalar_t>({val})')
    return line


def _vec4_replace(line, states, is_q4, idx_str, c_name, extra_inputs):
    """Replace macros for vec4 component processing."""
    for sname in states:
        load_pat_re = f'STATE_LOAD({sname}, idx)'
        if load_pat_re in line:
            if is_q4:
                line = line.replace(load_pat_re, f'{sname}_val_{c_name}')
            else:
                line = line.replace(load_pat_re, f'{sname}_v.{c_name}')

        store_pat = f'STATE_STORE({sname}, idx, '
        if store_pat in line:
            start = line.index(store_pat) + len(store_pat)
            depth = 1
            end = start
            while end < len(line) and depth > 0:
                if line[end] == '(':
                    depth += 1
                elif line[end] == ')':
                    depth -= 1
                end += 1
            val_expr = line[start:end-1]
            full_macro = line[line.index(f'STATE_STORE({sname}'):end]
            if is_q4:
                line = line.replace(full_macro, f'{sname}_val_{c_name} = {val_expr}')
            else:
                line = line.replace(full_macro, f'{sname}_v.{c_name} = {val_expr}')

    for ename in extra_inputs:
        line = line.replace(f'EXTRA_LOAD({ename}, idx)', f'{ename}_v.{c_name}')

    # Replace 'g' references carefully
    line = line.replace(' g;', f' g.{c_name};')
    line = line.replace(' g)', f' g.{c_name})')
    line = line.replace(' g ', f' g.{c_name} ')
    line = line.replace('(g ', f'(g.{c_name} ')
    line = line.replace(' g*', f' g.{c_name}*')
    line = line.replace('*g ', f'*g.{c_name} ')
    line = line.replace('*g)', f'*g.{c_name})')

    return line


def _vec4_replace_param(line, c_name):
    """Replace PARAM_LOAD/PARAM_STORE for vec4 component."""
    line = line.replace('PARAM_LOAD(idx)', f'p.{c_name}')
    import re
    m = re.search(r'PARAM_STORE\(idx,\s*(.+?)\)', line)
    if m:
        val = m.group(1)
        line = line.replace(m.group(0), f'p.{c_name} = {val}')
    return line


def _vec4_replace_grad(line, c_name, grad_name):
    """Replace GRAD_STORE for vec4 component."""
    import re
    m = re.search(r'GRAD_STORE\(idx,\s*(.+?)\)', line)
    if m:
        val = m.group(1)
        line = line.replace(m.group(0), f'{grad_name}4[i].{c_name} = {val}')
    return line


def _vec4_effective(expr, c_name, extra_inputs):
    """Convert effective_grad expression to vec4 component reference."""
    if not expr:
        return ''
    result = expr
    # Simple variable references
    for ename in extra_inputs:
        result = result.replace(ename, f'{ename}_v.{c_name}')
    # Replace standalone 'g' references
    result = result.replace('g', f'g.{c_name}')
    return result


def _gen_launcher(lines, optimizer_name, kern_name, spec, v):
    """Generate the C++ launcher function."""
    states = spec.get('states', {})
    scalars = spec.get('scalars', [])
    block_size = spec.get('block_size', 256)
    extra_inputs = spec.get('extra_inputs', {})
    grad_name = spec.get('grad_name', 'grad')
    is_vec4 = v['vec4']
    is_q4 = v['config4']
    is_moe = v['moe']

    lines.append(f'void launch_{kern_name}(')
    lines.append(f'    torch::Tensor param,')
    lines.append(f'    torch::Tensor {grad_name},')

    for sname, sconf in states.items():
        if is_q4:
            lines.append(f'    torch::Tensor {sname}_q,')
            if sconf['q4_format'] in ('int8_block8', 'int4_block32'):
                lines.append(f'    torch::Tensor {sname}_scales,')
        else:
            lines.append(f'    torch::Tensor {sname},')

    for ename in extra_inputs:
        lines.append(f'    torch::Tensor {ename},')

    if is_moe:
        lines.append(f'    torch::Tensor active_mask,')

    for sc in scalars:
        lines.append(f'    float {sc},')

    if is_q4:
        lines.append(f'    int global_step')
    else:
        # Remove trailing comma from last scalar
        if lines[-1].endswith(','):
            lines[-1] = lines[-1][:-1]

    lines.append(') {')
    lines.append(f'    const int N = param.numel();')
    lines.append(f'    if (N == 0) return;')
    lines.append('')

    if is_vec4:
        lines.append(f'    const int N4 = N / 4;')
        lines.append(f'    const int grid = (N4 + {block_size} - 1) / {block_size};')
        lines.append(f'    {kern_name}_kernel<<<grid, {block_size}>>>(')
        lines.append(f'        reinterpret_cast<float4*>(param.data_ptr<float>()),')
        lines.append(f'        reinterpret_cast<const float4*>({grad_name}.data_ptr<float>()),')
        for sname, sconf in states.items():
            if is_q4:
                if sconf['q4_format'] == 'bf16_sr':
                    lines.append(f'        reinterpret_cast<__nv_bfloat16*>({sname}_q.data_ptr()),')
                else:
                    lines.append(f'        {sname}_q.data_ptr<int8_t>(),')
                    lines.append(f'        {sname}_scales.data_ptr<float>(),')
            else:
                lines.append(f'        reinterpret_cast<float4*>({sname}.data_ptr<float>()),')
        for ename in extra_inputs:
            lines.append(f'        reinterpret_cast<const float4*>({ename}.data_ptr<float>()),')
        if is_moe:
            lines.append(f'        active_mask.data_ptr<bool>(),')
        for sc in scalars:
            lines.append(f'        {sc},')
        if is_q4:
            lines.append(f'        static_cast<unsigned>(global_step),')
        lines.append(f'        N4);')
    else:
        lines.append(f'    const int grid = (N + {block_size} - 1) / {block_size};')
        lines.append(f'    AT_DISPATCH_FLOATING_TYPES_AND2(')
        lines.append(f'        at::ScalarType::Half, at::ScalarType::BFloat16,')
        lines.append(f'        param.scalar_type(), "launch_{kern_name}", ([&] {{')
        lines.append(f'        {kern_name}_kernel<scalar_t><<<grid, {block_size}>>>(')
        lines.append(f'            param.data_ptr<scalar_t>(),')
        lines.append(f'            {grad_name}.data_ptr<scalar_t>(),')
        for sname, sconf in states.items():
            if is_q4:
                if sconf['q4_format'] == 'bf16_sr':
                    lines.append(f'            reinterpret_cast<__nv_bfloat16*>({sname}_q.data_ptr()),')
                else:
                    lines.append(f'            {sname}_q.data_ptr<int8_t>(),')
                    lines.append(f'            {sname}_scales.data_ptr<float>(),')
            else:
                lines.append(f'            {sname}.data_ptr<float>(),')
        for ename, econf in extra_inputs.items():
            ctype = econf['type'].replace('const ', '').replace('*', '').strip()
            if ctype == 'float':
                lines.append(f'            {ename}.data_ptr<float>(),')
            else:
                lines.append(f'            {ename}.data_ptr<scalar_t>(),')
        if is_moe:
            lines.append(f'            active_mask.data_ptr<bool>(),')
        for sc in scalars:
            lines.append(f'            {sc},')
        if is_q4:
            lines.append(f'            static_cast<unsigned>(global_step),')
        lines.append(f'            N);')
        lines.append(f'    }}));')

    lines.append('}')


def _generate_cpu_kernel(optimizer_name, spec, variant, kern_name, v):
    """Generate a CPU OpenMP kernel variant."""
    states = spec.get('states', {})
    scalars = spec.get('scalars', [])
    has_adam = spec.get('has_adam', False)
    is_q4 = v['config4']
    grad_name = spec.get('grad_name', 'grad')
    extra_inputs = spec.get('extra_inputs', {})

    lines = []
    lines.append(f'/* GENERATED by codegen/generate_kernels.py — DO NOT EDIT */')
    lines.append(f'/* CPU Optimizer: {optimizer_name}, Variant: {variant} */')
    lines.append('')
    lines.append('#include <cmath>')
    lines.append('#include <cstring>')
    lines.append('#include <algorithm>')
    lines.append('#ifdef _OPENMP')
    lines.append('#include <omp.h>')
    lines.append('#endif')
    lines.append('')

    if is_q4:
        # CPU stochastic rounding helpers
        lines.append('static inline unsigned cpu_philox_hash(unsigned key, unsigned salt) {')
        lines.append('    unsigned v = key * 2654435761u + salt * 2246822519u;')
        lines.append('    v ^= v >> 16; v *= 0x45d9f3bu; v ^= v >> 16;')
        lines.append('    return v;')
        lines.append('}')
        lines.append('')
        lines.append('static inline int8_t cpu_float_to_int8_sr(float val, float scale, unsigned rng) {')
        lines.append('    float scaled = val / std::fmax(scale, 1e-12f);')
        lines.append('    float tr = std::trunc(scaled);')
        lines.append('    float frac = std::fabs(scaled - tr);')
        lines.append('    float threshold = (float)(rng & 0xFFFF) / 65536.0f;')
        lines.append('    if (frac > threshold) tr += (scaled > 0) ? 1.0f : -1.0f;')
        lines.append('    return (int8_t)std::fmax(-127.0f, std::fmin(127.0f, tr));')
        lines.append('}')
        lines.append('')

    # Function signature
    lines.append(f'void {kern_name}(')
    lines.append(f'    float* param, const float* {grad_name},')
    for sname, sconf in states.items():
        if is_q4:
            lines.append(f'    int8_t* {sname}_q, float* {sname}_scales,')
        else:
            lines.append(f'    float* {sname},')
    for ename in extra_inputs:
        lines.append(f'    const float* {ename},')
    for sc in scalars:
        lines.append(f'    float {sc},')
    if is_q4:
        lines.append(f'    unsigned global_step,')
    lines.append(f'    int N')
    lines.append(') {')
    lines.append(f'    #pragma omp parallel for schedule(static)')
    lines.append(f'    for (int idx = 0; idx < N; idx++) {{')
    lines.append(f'        const float g = {grad_name}[idx];')

    if has_adam:
        effective_grad = spec.get('effective_grad', 'g')
        pre_adam = spec.get('pre_adam_math', '')
        effective_grad_sq = spec.get('effective_grad_sq', f'{effective_grad} * {effective_grad}')
        adam_denom_extra = spec.get('adam_denom_extra', '')
        adam_decay_extra = spec.get('adam_decay_extra', '')

        if pre_adam:
            for line in pre_adam.strip().split('\n'):
                l = line.strip()
                if not l:
                    continue
                l = _cpu_replace_state_macros(l, states, is_q4)
                for ename in extra_inputs:
                    l = l.replace(f'EXTRA_LOAD({ename}, idx)', f'{ename}[idx]')
                lines.append(f'        {l}')

        # Load moments
        for sname in ['exp_avg', 'exp_avg_sq']:
            if sname in states:
                if is_q4:
                    lines.append(f'        float {sname}_val = (float){sname}_q[idx] * {sname}_scales[idx / 8];')
                else:
                    lines.append(f'        float {sname}_val = {sname}[idx];')

        lines.append(f'        float ea = beta1 * exp_avg_val + (1.0f - beta1) * {effective_grad};')
        lines.append(f'        float easq = beta2 * exp_avg_sq_val + (1.0f - beta2) * {effective_grad_sq};')

        # Store moments
        for sname in ['exp_avg', 'exp_avg_sq']:
            if sname in states:
                val = 'ea' if sname == 'exp_avg' else 'easq'
                if is_q4:
                    lines.append(f'        {{')
                    lines.append(f'            float _scale = std::fmax(std::fabs({val}), 1e-12f) / 127.0f;')
                    lines.append(f'            unsigned _rng = cpu_philox_hash(global_step, (unsigned)idx);')
                    lines.append(f'            {sname}_q[idx] = cpu_float_to_int8_sr({val}, _scale, _rng);')
                    lines.append(f'            if (idx % 8 == 0) {sname}_scales[idx / 8] = _scale;')
                    lines.append(f'        }}')
                else:
                    lines.append(f'        {sname}[idx] = {val};')

        lines.append(f'        float step_size = lr / bc1;')
        lines.append(f'        float denom = std::sqrt(easq / bc2) + {adam_denom_extra}eps;')
        lines.append(f'        param[idx] = param[idx] * (1.0f - lr * {adam_decay_extra}weight_decay) - step_size * ea / denom;')

    else:
        update_math = spec.get('update_math', '')
        if update_math:
            for line in update_math.strip().split('\n'):
                l = line.strip()
                if not l:
                    continue
                l = _cpu_replace_state_macros(l, states, is_q4)
                l = l.replace('PARAM_LOAD(idx)', 'param[idx]')
                import re
                m = re.search(r'PARAM_STORE\(idx,\s*(.+?)\)', l)
                if m:
                    val = m.group(1)
                    l = l.replace(m.group(0), f'param[idx] = {val}')
                m = re.search(r'GRAD_STORE\(idx,\s*(.+?)\)', l)
                if m:
                    val = m.group(1)
                    l = l.replace(m.group(0), f'{grad_name}[idx] = {val}')
                for ename in extra_inputs:
                    l = l.replace(f'EXTRA_LOAD({ename}, idx)', f'{ename}[idx]')
                # Remove CUDA-specific casts
                l = l.replace('static_cast<scalar_t>', '')
                l = l.replace('static_cast<float>', '')
                lines.append(f'        {l}')

    lines.append(f'    }}')
    lines.append(f'}}')
    return '\n'.join(lines) + '\n'


def _cpu_replace_state_macros(line, states, is_q4):
    """Replace STATE_LOAD/STATE_STORE macros for CPU code."""
    for sname, sconf in states.items():
        load_pat = f'STATE_LOAD({sname}, idx)'
        if load_pat in line:
            if is_q4:
                line = line.replace(load_pat, f'((float){sname}_q[idx] * {sname}_scales[idx / 8])')
            else:
                line = line.replace(load_pat, f'{sname}[idx]')

        store_pat = f'STATE_STORE({sname}, idx, '
        if store_pat in line:
            start = line.index(store_pat) + len(store_pat)
            depth = 1
            end = start
            while end < len(line) and depth > 0:
                if line[end] == '(':
                    depth += 1
                elif line[end] == ')':
                    depth -= 1
                end += 1
            val_expr = line[start:end-1]
            full_macro = line[line.index(f'STATE_STORE({sname}'):end]
            if is_q4:
                replacement = f'{sname}_q[idx] = cpu_float_to_int8_sr({val_expr}, std::fmax(std::fabs({val_expr}), 1e-12f) / 127.0f, cpu_philox_hash(global_step, (unsigned)idx))'
            else:
                replacement = f'{sname}[idx] = {val_expr}'
            line = line.replace(full_macro, replacement)
    return line


# ═══════════════════════════════════════════════════════════════════════
#  Main: read specs, generate all kernels, write to output directory
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Generate SuperGrok v2 CUDA kernels')
    parser.add_argument('--output', default='csrc/cuda/generated/',
                        help='Output directory for generated .cu files')
    parser.add_argument('--specs', default='codegen/kernel_specs.yaml',
                        help='Path to kernel specifications YAML')
    parser.add_argument('--cpu-output', default='csrc/cpu/generated/',
                        help='Output directory for generated CPU .cpp files')
    args = parser.parse_args()

    # Load specs
    with open(args.specs) as f:
        specs = yaml.safe_load(f)

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.cpu_output, exist_ok=True)

    total_generated = 0
    gpu_count = 0
    cpu_count = 0

    for opt_name, spec in specs.items():
        variants = spec.get('variants', {})

        # GPU variants
        gpu_variants = variants.get('gpu', [])
        if gpu_variants:
            gpu_kernels = []
            for variant in gpu_variants:
                code = generate_elementwise_kernel(opt_name, spec, variant)
                gpu_kernels.append(code)

            # Write all GPU variants for this optimizer to one .cu file
            output_path = os.path.join(args.output, f'{opt_name}_generated.cu')
            with open(output_path, 'w') as f:
                f.write(f'/* GENERATED by codegen/generate_kernels.py — DO NOT EDIT */\n')
                f.write(f'/* Optimizer: {opt_name} — {len(gpu_variants)} GPU variants */\n\n')
                f.write('#include <torch/extension.h>\n')
                f.write('#include "platform.h"\n')
                f.write('#include "utils.cuh"\n\n')

                # Write Philox RNG only once if any Q4 variant exists
                has_q4 = any(parse_variant(v)['config4'] for v in gpu_variants)
                if has_q4:
                    f.write('// Philox RNG (stateless)\n')
                    f.write('__device__ __forceinline__ unsigned philox_hash_gen(unsigned key, unsigned salt) {\n')
                    f.write('    unsigned v = key * 2654435761u + salt * 2246822519u;\n')
                    f.write('    v ^= v >> 16; v *= 0x45d9f3bu; v ^= v >> 16;\n')
                    f.write('    return v;\n')
                    f.write('}\n\n')
                    f.write('__device__ __forceinline__ int8_t float_to_int4_stochastic_gen(\n')
                    f.write('    float val, float scale, unsigned rand_bits) {\n')
                    f.write('    float scaled = val / fmaxf(scale, 1e-12f);\n')
                    f.write('    float tr = truncf(scaled);\n')
                    f.write('    float frac = fabsf(scaled - tr);\n')
                    f.write('    float threshold = (float)(rand_bits & 0xFFFF) / 65536.0f;\n')
                    f.write('    if (frac > threshold) tr += (scaled > 0) ? 1.0f : -1.0f;\n')
                    f.write('    return (int8_t)fmaxf(-7.0f, fminf(7.0f, tr));\n')
                    f.write('}\n\n')

                # Write each variant's kernel (strip duplicate headers)
                for code in gpu_kernels:
                    # Remove duplicate includes and helper functions
                    cleaned = []
                    skip = False
                    for line in code.split('\n'):
                        if line.startswith('#include') or line.startswith('/* GENERATED'):
                            continue
                        if 'philox_hash' in line and '__device__' in line:
                            skip = True
                        if skip and line.strip() == '}':
                            skip = False
                            continue
                        if skip:
                            continue
                        if 'float_to_int4_stochastic' in line and '__device__' in line:
                            skip = True
                            continue
                        # Remap philox_hash → philox_hash_gen to avoid collisions with utils.cuh
                        line = line.replace('philox_hash(', 'philox_hash_gen(')
                        line = line.replace('float_to_int4_stochastic(', 'float_to_int4_stochastic_gen(')
                        cleaned.append(line)
                    f.write('\n'.join(cleaned))
                    f.write('\n\n')

            gpu_count += len(gpu_variants)
            total_generated += len(gpu_variants)
            print(f'  {opt_name}: {len(gpu_variants)} GPU variants → {output_path}')

        # CPU variants
        cpu_variants = variants.get('cpu', [])
        if cpu_variants:
            for variant in cpu_variants:
                code = generate_elementwise_kernel(opt_name, spec, variant)
                v = parse_variant(variant)
                suffix = kernel_suffix(v)
                output_path = os.path.join(args.cpu_output,
                    f'{opt_name}_{suffix}_generated.cpp')
                with open(output_path, 'w') as f:
                    f.write(code)
                cpu_count += 1
                total_generated += 1

            print(f'  {opt_name}: {len(cpu_variants)} CPU variants → {args.cpu_output}')

    print(f'\nTotal generated: {total_generated} kernels ({gpu_count} GPU, {cpu_count} CPU)')
    return total_generated


if __name__ == '__main__':
    total = main()
    print(f'\nDone. Generated {total} kernel variants.')
