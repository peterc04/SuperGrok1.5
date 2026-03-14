"""
Mamba-3 + 4-Head PEER + Per-Element GRU Meta-Net for SuperGrok v2

Architecture per optimizer step, per parameter:
  1. SORT: Sort N gradient elements by magnitude (creates meaningful sequence)
  2. MAMBA-3 BIDIRECTIONAL SCAN:
     - Forward scan: h_fwd_i encodes all elements with smaller |gradient|
     - Backward scan: h_bwd_i encodes all elements with larger |gradient|
     - Provides cross-element awareness via selective state space dynamics
     - Complex-valued state captures periodic/oscillatory gradient patterns
  3. PER-ELEMENT GRU (tiny, 4-dim):
     - Input: [grad_i, sharpness_i, scan_output_i]
     - Persistent state across optimizer steps (temporal memory)
  4. 4-HEAD PEER ROUTING:
     - Each head: query from [gru_state, scan_context, grad, sharpness]
     - Product-key lookup selects 1 of num_experts per head
     - 4 experts activated per element, outputs summed
  5. EXPERT MLP:
     - Selected expert transforms gradient: 1 -> hidden -> 1
     - 144 experts default, hidden=16 default
  6. DYNAMIC EXPERT RECYCLING (every recycle_interval steps):
     - Count activations per expert
     - Dead experts (< threshold activations): clone top expert + noise
     - Reset dead expert's product key to random direction
  7. SKIP CONNECTION:
     - smart_grad = grad + rescale * sum(expert_outputs)

  Complexity: O(N log N) sort + O(N * d_state) scan + O(N * H * sqrt(E)) routing
  All components trained via bilevel optimization.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

try:
    from grokking_optimizers import _ops as _ops
    _HAS_CUDA_BACKWARD = hasattr(_ops, 'supergrok2_bilevel_fwd_save')
except ImportError:
    _HAS_CUDA_BACKWARD = False


class Mamba3ScanBlock(nn.Module):
    """Simplified Mamba-3 selective scan for per-parameter gradient processing.

    Uses:
      - Trapezoidal discretization (Mamba-3 style, more accurate than Euler)
      - Complex-valued state via real-valued RoPE equivalence
      - Selective gating on B, C, dt (input-dependent dynamics)

    This is a MINIMAL Mamba-3 implementation optimized for the meta-net use case:
      - Small d_model (8), small d_state (16)
      - Short sequences by LLM standards but large by optimizer standards (N=65K)
      - No conv1d (unnecessary for sorted gradient sequences)
    """

    def __init__(self, d_model: int = 8, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection: d_model -> 2*d_inner (x and z branches)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # SSM parameters (selective/input-dependent)
        # dt projection: d_inner -> d_inner (controls how much to update state)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        # B and C projections: d_inner -> d_state
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)

        # A: diagonal state transition (log-space for stability)
        # Initialize as negative real values (decaying dynamics)
        self.A_log = nn.Parameter(
            torch.log(torch.linspace(1, d_state, d_state)).unsqueeze(0).expand(self.d_inner, -1))

        # D: skip connection within SSM
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # RoPE-equivalent phase for complex dynamics (Mamba-3)
        # Learnable per-state frequencies
        self.rope_freq = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.01)

        # Output projection: d_inner -> d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, initial_state: Optional[torch.Tensor] = None):
        """
        Args:
            x: [N, d_model] -- sorted gradient features
            initial_state: [d_inner, d_state] or None
        Returns:
            output: [N, d_model]
            final_state: [d_inner, d_state]
        """
        N = x.shape[0]

        # Project input
        xz = self.in_proj(x)  # [N, 2*d_inner]
        x_branch, z = xz.chunk(2, dim=-1)  # each [N, d_inner]

        # Selective parameters (input-dependent)
        dt = torch.softplus(self.dt_proj(x_branch))  # [N, d_inner] -- positive
        B = self.B_proj(x_branch)                       # [N, d_state]
        C = self.C_proj(x_branch)                       # [N, d_state]

        # State transition: A = -exp(A_log) (always negative = decaying)
        A = -torch.exp(self.A_log)  # [d_inner, d_state]

        # RoPE phase rotation for complex dynamics
        phase = self.rope_freq  # [d_inner, d_state]

        # Selective scan (sequential -- Python reference, CUDA kernel in Phase C)
        if initial_state is not None:
            h = initial_state.clone()
        else:
            h = torch.zeros(self.d_inner, self.d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for i in range(N):
            # Trapezoidal discretization (Mamba-3):
            # h_new = (1 + dt*A/2)/(1 - dt*A/2) * h + dt * B * x
            # Simplified for diagonal A:
            dt_i = dt[i].unsqueeze(-1)  # [d_inner, 1]
            A_bar = (1.0 + dt_i * A / 2.0) / (1.0 - dt_i * A / 2.0 + 1e-8)
            B_bar = dt_i * B[i].unsqueeze(0)  # [d_inner, d_state]

            # Complex rotation via RoPE-equivalent
            cos_phase = torch.cos(dt_i * phase)
            sin_phase = torch.sin(dt_i * phase)
            # Apply rotation to state (pairs of dimensions act as real/imag)
            h_rot = h * cos_phase - torch.roll(h, 1, dims=-1) * sin_phase

            # State update
            h = A_bar * h_rot + B_bar * x_branch[i].unsqueeze(-1)

            # Output
            y_i = (h * C[i].unsqueeze(0)).sum(dim=-1)  # [d_inner]
            outputs.append(y_i)

        y = torch.stack(outputs, dim=0)  # [N, d_inner]

        # Gated output
        y = y * torch.silu(z)  # [N, d_inner]

        # Skip connection within SSM
        y = y + self.D.unsqueeze(0) * x_branch

        # Project back to d_model
        output = self.out_proj(y)  # [N, d_model]

        return output, h


class MiniGRU(nn.Module):
    """Tiny per-element GRU for temporal memory across optimizer steps.

    4-dimensional hidden state per gradient element.
    Input: [grad, sharpness, scan_context (d_model floats)]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        # GRU gates
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)  # update gate
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)  # reset gate
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)  # candidate

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """
        Args:
            x: [N, input_dim]
            h: [N, hidden_dim]
        Returns:
            h_new: [N, hidden_dim]
        """
        xh = torch.cat([x, h], dim=-1)
        z = torch.sigmoid(self.W_z(xh))      # update gate
        r = torch.sigmoid(self.W_r(xh))      # reset gate
        xrh = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.W_h(xrh))  # candidate
        h_new = (1 - z) * h + z * h_tilde
        return h_new


class Mamba3PEERMetaNet(nn.Module):
    """Mamba-3 + 4-Head PEER + Per-Element GRU meta-network.

    Args:
        d_model: Internal dimension (default: 8)
        d_state: Mamba state dimension (default: 16)
        mamba_expand: Mamba expansion factor (default: 2)
        num_peer_heads: Number of PEER routing heads (default: 4)
        num_experts: Total experts in pool (default: 144, must be perfect square)
        expert_hidden: Hidden dim per expert MLP (default: 16)
        gru_hidden: Per-element GRU hidden dim (default: 4)
        rescale: Skip connection scale (default: 0.1)
        recycle_interval: Steps between dead expert recycling (default: 100)
        recycle_threshold: Min activation fraction to stay alive (default: 0.001)
    """

    def __init__(
        self,
        d_model: int = 8,
        d_state: int = 16,
        mamba_expand: int = 2,
        num_peer_heads: int = 4,
        num_experts: int = 144,
        expert_hidden: int = 16,
        gru_hidden: int = 4,
        rescale: float = 0.1,
        recycle_interval: int = 100,
        recycle_threshold: float = 0.001,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_peer_heads = num_peer_heads
        self.num_experts = num_experts
        self.expert_hidden = expert_hidden
        self.gru_hidden = gru_hidden
        self.rescale = rescale
        self.recycle_interval = recycle_interval
        self.recycle_threshold = recycle_threshold

        self.pk_dim = int(math.sqrt(num_experts))
        assert self.pk_dim * self.pk_dim == num_experts, \
            f"num_experts must be perfect square, got {num_experts}"

        # -- Input projection
        self.input_proj = nn.Linear(2, d_model, bias=True)

        # -- Bidirectional Mamba-3
        self.mamba_fwd = Mamba3ScanBlock(d_model, d_state, mamba_expand)
        self.mamba_bwd = Mamba3ScanBlock(d_model, d_state, mamba_expand)

        # -- Per-element GRU (temporal memory)
        gru_input_dim = 2 + 2 * d_model  # [grad, sharp, fwd_context, bwd_context]
        self.gru = MiniGRU(gru_input_dim, gru_hidden)

        # -- Multi-head PEER routing
        peer_input_dim = gru_hidden + 2 * d_model + 2  # [gru_state, fwd_ctx, bwd_ctx, grad, sharp]
        self.peer_queries = nn.ModuleList([
            nn.Linear(peer_input_dim, d_model, bias=False)
            for _ in range(num_peer_heads)
        ])
        self.product_keys_A = nn.ParameterList([
            nn.Parameter(torch.randn(self.pk_dim, d_model // 2) * 0.02)
            for _ in range(num_peer_heads)
        ])
        self.product_keys_B = nn.ParameterList([
            nn.Parameter(torch.randn(self.pk_dim, d_model // 2) * 0.02)
            for _ in range(num_peer_heads)
        ])

        # -- Expert pool (shared across heads)
        self.expert_W1 = nn.Parameter(torch.randn(num_experts, expert_hidden, 1) * 0.02)
        self.expert_b1 = nn.Parameter(torch.zeros(num_experts, expert_hidden))
        self.expert_W2 = nn.Parameter(torch.randn(num_experts, 1, expert_hidden) * 0.02)
        self.expert_b2 = nn.Parameter(torch.zeros(num_experts, 1))

        # -- Expert activation tracking (not a parameter)
        self.register_buffer('expert_counts', torch.zeros(num_experts, dtype=torch.long))
        self.register_buffer('step_counter', torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        grad: torch.Tensor,
        sharpness: torch.Tensor,
        gru_state: torch.Tensor,
        mamba_fwd_state: Optional[torch.Tensor] = None,
        mamba_bwd_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            grad: [N] flattened gradient
            sharpness: [N] flattened sharpness
            gru_state: [N, gru_hidden] persistent per-element state
            mamba_fwd_state: [d_inner, d_state] or None
            mamba_bwd_state: [d_inner, d_state] or None
        Returns:
            smart_grad: [N]
            new_gru_state: [N, gru_hidden]
            new_mamba_fwd_state: [d_inner, d_state]
            new_mamba_bwd_state: [d_inner, d_state]
        """
        N = grad.numel()
        g = grad.reshape(-1).float()
        s = sharpness.reshape(-1).float()

        # 1. Sort by magnitude (creates meaningful sequence for Mamba)
        sort_idx = g.abs().argsort()
        g_sorted = g[sort_idx]
        s_sorted = s[sort_idx]

        # 2. Input projection
        inp = torch.stack([g_sorted, s_sorted], dim=-1)  # [N, 2]
        x = self.input_proj(inp)  # [N, d_model]

        # 3. Bidirectional Mamba-3 scan
        fwd_out, new_fwd_state = self.mamba_fwd(x, mamba_fwd_state)       # [N, d_model]
        bwd_out, new_bwd_state = self.mamba_bwd(x.flip(0), mamba_bwd_state)
        bwd_out = bwd_out.flip(0)  # reverse back

        # 4. Unsort to original order
        unsort_idx = sort_idx.argsort()
        fwd_ctx = fwd_out[unsort_idx]   # [N, d_model]
        bwd_ctx = bwd_out[unsort_idx]   # [N, d_model]

        # 5. Per-element GRU update
        gru_input = torch.cat([
            g.unsqueeze(-1), s.unsqueeze(-1),  # [N, 1] each
            fwd_ctx, bwd_ctx                     # [N, d_model] each
        ], dim=-1)  # [N, 2 + 2*d_model]
        new_gru = self.gru(gru_input, gru_state.float())  # [N, gru_hidden]

        # 6. Multi-head PEER routing
        peer_input = torch.cat([
            new_gru, fwd_ctx, bwd_ctx,
            g.unsqueeze(-1), s.unsqueeze(-1)
        ], dim=-1)  # [N, gru_hidden + 2*d_model + 2]

        total_expert_out = torch.zeros(N, 1, device=grad.device, dtype=torch.float32)

        for h in range(self.num_peer_heads):
            query = self.peer_queries[h](peer_input)  # [N, d_model]
            q_a = query[:, :self.d_model // 2]
            q_b = query[:, self.d_model // 2:]

            idx_a = (q_a @ self.product_keys_A[h].T).argmax(dim=-1)  # [N]
            idx_b = (q_b @ self.product_keys_B[h].T).argmax(dim=-1)  # [N]
            expert_idx = idx_a * self.pk_dim + idx_b  # [N]

            # Track activations (no grad needed)
            if self.training:
                with torch.no_grad():
                    self.expert_counts.scatter_add_(
                        0, expert_idx,
                        torch.ones_like(expert_idx, dtype=torch.long))

            # Expert evaluation
            W1 = self.expert_W1[expert_idx]  # [N, expert_hidden, 1]
            b1 = self.expert_b1[expert_idx]  # [N, expert_hidden]
            W2 = self.expert_W2[expert_idx]  # [N, 1, expert_hidden]
            b2 = self.expert_b2[expert_idx]  # [N, 1]

            z = torch.relu(torch.bmm(W1, g.unsqueeze(-1).unsqueeze(-1)).squeeze(-1) + b1)
            out = torch.bmm(W2, z.unsqueeze(-1)).squeeze(-1) + b2  # [N, 1]
            total_expert_out = total_expert_out + out

        # Average over heads
        total_expert_out = total_expert_out / self.num_peer_heads

        # 7. Dynamic expert recycling
        if self.training:
            self.step_counter += 1
            if self.step_counter.item() % self.recycle_interval == 0:
                self._recycle_dead_experts()

        # 8. Skip connection
        smart_grad = (g.unsqueeze(-1) + self.rescale * total_expert_out).squeeze(-1)
        smart_grad = smart_grad.reshape(grad.shape).to(grad.dtype)

        return smart_grad, new_gru, new_fwd_state, new_bwd_state

    def forward_for_bilevel(
        self, grad, sharpness, gru_state,
        mamba_fwd_state=None, mamba_bwd_state=None,
    ):
        """Differentiable forward with top-k sparse soft PEER routing."""
        N = grad.numel()
        g = grad.reshape(-1).float()
        s = sharpness.reshape(-1).float()

        sort_idx = g.abs().argsort()
        g_sorted = g[sort_idx]
        s_sorted = s[sort_idx]

        inp = torch.stack([g_sorted, s_sorted], dim=-1)
        x = self.input_proj(inp)

        fwd_out, new_fwd = self.mamba_fwd(x, mamba_fwd_state)
        bwd_out, new_bwd = self.mamba_bwd(x.flip(0), mamba_bwd_state)
        bwd_out = bwd_out.flip(0)

        unsort_idx = sort_idx.argsort()
        fwd_ctx = fwd_out[unsort_idx]
        bwd_ctx = bwd_out[unsort_idx]

        gru_input = torch.cat([g.unsqueeze(-1), s.unsqueeze(-1), fwd_ctx, bwd_ctx], dim=-1)
        new_gru = self.gru(gru_input, gru_state.float())

        peer_input = torch.cat([new_gru, fwd_ctx, bwd_ctx, g.unsqueeze(-1), s.unsqueeze(-1)], dim=-1)

        total_expert_out = torch.zeros(N, 1, device=grad.device, dtype=torch.float32)
        topk = 4  # top-4 per sub-key per head

        for h in range(self.num_peer_heads):
            query = self.peer_queries[h](peer_input)
            q_a = query[:, :self.d_model // 2]
            q_b = query[:, self.d_model // 2:]

            scores_a = q_a @ self.product_keys_A[h].T  # [N, pk_dim]
            scores_b = q_b @ self.product_keys_B[h].T  # [N, pk_dim]

            top_a_vals, top_a_idx = scores_a.topk(topk, dim=-1)
            top_b_vals, top_b_idx = scores_b.topk(topk, dim=-1)

            soft_a = torch.softmax(top_a_vals * 10.0, dim=-1)
            soft_b = torch.softmax(top_b_vals * 10.0, dim=-1)

            expert_indices = (top_a_idx.unsqueeze(2) * self.pk_dim + top_b_idx.unsqueeze(1)).reshape(N, -1)
            routing_weights = (soft_a.unsqueeze(2) * soft_b.unsqueeze(1)).reshape(N, -1)

            W1 = self.expert_W1[expert_indices]
            b1 = self.expert_b1[expert_indices]
            W2 = self.expert_W2[expert_indices]
            b2 = self.expert_b2[expert_indices]

            num_active = topk * topk  # 16
            g_exp = g.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(-1, num_active, -1, -1)
            z = torch.relu(torch.matmul(W1, g_exp).squeeze(-1) + b1)
            out = torch.matmul(W2, z.unsqueeze(-1)).squeeze(-1).squeeze(-1) + b2.squeeze(-1)
            head_out = (routing_weights * out).sum(dim=1, keepdim=True)
            total_expert_out = total_expert_out + head_out

        total_expert_out = total_expert_out / self.num_peer_heads
        smart_grad = (g.unsqueeze(-1) + self.rescale * total_expert_out).squeeze(-1)
        return smart_grad.reshape(grad.shape).to(grad.dtype), new_gru, new_fwd, new_bwd

    @torch.no_grad()
    def _recycle_dead_experts(self):
        """Replace dead experts with mutated clones of top performers."""
        total_activations = self.expert_counts.sum().item()
        if total_activations == 0:
            return

        fractions = self.expert_counts.float() / total_activations
        dead_mask = fractions < self.recycle_threshold

        if not dead_mask.any():
            self.expert_counts.zero_()
            return

        # Find top-performing expert (most activations = most trusted by router)
        top_expert = self.expert_counts.argmax().item()
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]

        for idx in dead_indices:
            i = idx.item()
            # Clone top expert weights + small noise
            noise_scale = 0.01
            self.expert_W1.data[i] = self.expert_W1.data[top_expert] + \
                noise_scale * torch.randn_like(self.expert_W1.data[i])
            self.expert_b1.data[i] = self.expert_b1.data[top_expert] + \
                noise_scale * torch.randn_like(self.expert_b1.data[i])
            self.expert_W2.data[i] = self.expert_W2.data[top_expert] + \
                noise_scale * torch.randn_like(self.expert_W2.data[i])
            self.expert_b2.data[i] = self.expert_b2.data[top_expert] + \
                noise_scale * torch.randn_like(self.expert_b2.data[i])

            # Only randomize a product key if ALL experts sharing it are dead
            a_idx = i // self.pk_dim
            b_idx = i % self.pk_dim

            # Check if all experts in row a_idx are dead
            row_start = a_idx * self.pk_dim
            row_end = row_start + self.pk_dim
            if dead_mask[row_start:row_end].all():
                for h in range(self.num_peer_heads):
                    self.product_keys_A[h].data[a_idx] = torch.randn_like(
                        self.product_keys_A[h].data[a_idx]) * 0.02

            # Check if all experts in column b_idx are dead
            col_indices = torch.arange(0, self.num_experts, self.pk_dim,
                                       device=dead_mask.device) + b_idx
            col_indices = col_indices[col_indices < self.num_experts]
            if dead_mask[col_indices].all():
                for h in range(self.num_peer_heads):
                    self.product_keys_B[h].data[b_idx] = torch.randn_like(
                        self.product_keys_B[h].data[b_idx]) * 0.02

        # Reset counters
        self.expert_counts.zero_()

    @property
    def has_cuda_bilevel(self):
        """Whether CUDA bilevel backward kernels are available."""
        return _HAS_CUDA_BACKWARD and next(self.parameters()).is_cuda

    def forward_for_bilevel_cuda(
        self, grad, sharpness, gru_state,
        mamba_fwd_state=None, mamba_bwd_state=None,
    ):
        """Bilevel forward with CUDA backward kernels available.

        Uses the Python forward path (which is differentiable via autograd)
        for the forward pass. The CUDA backward kernels are registered in
        the C++ extension (_ops.supergrok2_bilevel_backward) and can be
        used for manual gradient computation when building a fully CUDA
        bilevel pipeline.

        Falls back to forward_for_bilevel when CUDA is not available.
        """
        return self.forward_for_bilevel(
            grad, sharpness, gru_state, mamba_fwd_state, mamba_bwd_state)

    def get_weights(self):
        """Extract all weights for CUDA kernel (Phase C)."""
        return {
            # Input proj
            'input_proj_W': self.input_proj.weight.detach().float().contiguous(),
            'input_proj_b': self.input_proj.bias.detach().float().contiguous(),
            # Mamba forward
            'mamba_fwd_in_proj': self.mamba_fwd.in_proj.weight.detach().float().contiguous(),
            'mamba_fwd_dt_proj_W': self.mamba_fwd.dt_proj.weight.detach().float().contiguous(),
            'mamba_fwd_dt_proj_b': self.mamba_fwd.dt_proj.bias.detach().float().contiguous(),
            'mamba_fwd_B_proj': self.mamba_fwd.B_proj.weight.detach().float().contiguous(),
            'mamba_fwd_C_proj': self.mamba_fwd.C_proj.weight.detach().float().contiguous(),
            'mamba_fwd_A_log': self.mamba_fwd.A_log.detach().float().contiguous(),
            'mamba_fwd_D': self.mamba_fwd.D.detach().float().contiguous(),
            'mamba_fwd_rope_freq': self.mamba_fwd.rope_freq.detach().float().contiguous(),
            'mamba_fwd_out_proj': self.mamba_fwd.out_proj.weight.detach().float().contiguous(),
            # Mamba backward (same structure)
            'mamba_bwd_in_proj': self.mamba_bwd.in_proj.weight.detach().float().contiguous(),
            'mamba_bwd_dt_proj_W': self.mamba_bwd.dt_proj.weight.detach().float().contiguous(),
            'mamba_bwd_dt_proj_b': self.mamba_bwd.dt_proj.bias.detach().float().contiguous(),
            'mamba_bwd_B_proj': self.mamba_bwd.B_proj.weight.detach().float().contiguous(),
            'mamba_bwd_C_proj': self.mamba_bwd.C_proj.weight.detach().float().contiguous(),
            'mamba_bwd_A_log': self.mamba_bwd.A_log.detach().float().contiguous(),
            'mamba_bwd_D': self.mamba_bwd.D.detach().float().contiguous(),
            'mamba_bwd_rope_freq': self.mamba_bwd.rope_freq.detach().float().contiguous(),
            'mamba_bwd_out_proj': self.mamba_bwd.out_proj.weight.detach().float().contiguous(),
            # GRU
            'gru_W_z': self.gru.W_z.weight.detach().float().contiguous(),
            'gru_b_z': self.gru.W_z.bias.detach().float().contiguous(),
            'gru_W_r': self.gru.W_r.weight.detach().float().contiguous(),
            'gru_b_r': self.gru.W_r.bias.detach().float().contiguous(),
            'gru_W_h': self.gru.W_h.weight.detach().float().contiguous(),
            'gru_b_h': self.gru.W_h.bias.detach().float().contiguous(),
            # PEER (per head)
            'peer_queries': [q.weight.detach().float().contiguous() for q in self.peer_queries],
            'product_keys_A': [k.detach().float().contiguous() for k in self.product_keys_A],
            'product_keys_B': [k.detach().float().contiguous() for k in self.product_keys_B],
            # Experts
            'expert_W1': self.expert_W1.detach().float().contiguous(),
            'expert_b1': self.expert_b1.detach().float().contiguous(),
            'expert_W2': self.expert_W2.detach().float().contiguous(),
            'expert_b2': self.expert_b2.detach().float().contiguous(),
            # Scalars
            'rescale': self.rescale,
            'd_model': self.d_model,
            'd_state': self.d_state,
            'd_inner': self.mamba_fwd.d_inner,
            'pk_dim': self.pk_dim,
            'expert_hidden': self.expert_hidden,
            'gru_hidden': self.gru_hidden,
            'num_peer_heads': self.num_peer_heads,
            'num_experts': self.num_experts,
        }


    # CUDA backward kernels for bilevel are registered via pybind11:
    #   _ops.supergrok2_bilevel_fwd_save(...)  -- forward scan with state saving
    #   _ops.supergrok2_bilevel_backward(...)  -- full backward through meta-net
    # These can be used to build a fully CUDA bilevel pipeline when the
    # Python autograd path becomes a bottleneck.
