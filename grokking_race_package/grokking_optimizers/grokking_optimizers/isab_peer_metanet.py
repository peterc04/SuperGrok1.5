"""
ISAB + PEER + Recurrent Meta-Net for SuperGrok v2

Architecture per step per parameter:
  Kernel 1 (reduction): Elements update M inducing points via cross-attention
  Kernel 2 (fused per-element):
    1. Recurrent state update: h = tanh(W_h @ h + W_x @ [grad, sharpness])
    2. Cross-attend to updated inducing points for global context
    3. PEER product-key routing → select expert
    4. Expert MLP transforms gradient

  Complexity: O(N × M × d) — linear in N.
  All components trained via bilevel optimization.
"""

import math
import torch
import torch.nn as nn


class ISABPEERMetaNet(nn.Module):
    def __init__(
        self,
        num_inducing: int = 16,
        d_model: int = 8,
        num_peer_experts: int = 1024,  # must be perfect square
        expert_hidden: int = 4,
        recurrent_dim: int = 8,
        rescale: float = 0.1,
    ):
        super().__init__()
        self.num_inducing = num_inducing
        self.d_model = d_model
        self.num_peer_experts = num_peer_experts
        self.expert_hidden = expert_hidden
        self.recurrent_dim = recurrent_dim
        self.rescale = rescale

        self.pk_dim = int(math.sqrt(num_peer_experts))
        assert self.pk_dim * self.pk_dim == num_peer_experts

        # Inducing points (learned)
        self.inducing_points = nn.Parameter(torch.randn(num_inducing, d_model) * 0.02)

        # ISAB projections
        self.input_proj = nn.Linear(2, d_model, bias=True)
        self.induce_q = nn.Linear(d_model, d_model, bias=False)
        self.induce_k = nn.Linear(d_model, d_model, bias=False)
        self.induce_v = nn.Linear(d_model, d_model, bias=False)
        self.read_q = nn.Linear(d_model, d_model, bias=False)

        # Recurrent state
        self.W_h = nn.Linear(recurrent_dim, recurrent_dim, bias=False)
        self.W_x = nn.Linear(2, recurrent_dim, bias=True)

        # PEER product keys
        peer_input_dim = recurrent_dim + d_model + 2
        self.peer_query = nn.Linear(peer_input_dim, d_model, bias=False)
        self.product_keys_A = nn.Parameter(torch.randn(self.pk_dim, d_model // 2) * 0.02)
        self.product_keys_B = nn.Parameter(torch.randn(self.pk_dim, d_model // 2) * 0.02)

        # Experts: tiny MLPs stored as batched weights
        self.expert_W1 = nn.Parameter(torch.randn(num_peer_experts, expert_hidden, 1) * 0.02)
        self.expert_b1 = nn.Parameter(torch.zeros(num_peer_experts, expert_hidden))
        self.expert_W2 = nn.Parameter(torch.randn(num_peer_experts, 1, expert_hidden) * 0.02)
        self.expert_b2 = nn.Parameter(torch.zeros(num_peer_experts, 1))

    def forward(self, grad, sharpness, recurrent_state):
        """
        Args:
            grad: [N] flattened gradient
            sharpness: [N] flattened sharpness
            recurrent_state: [N, recurrent_dim] persistent state
        Returns:
            smart_grad: [N], new_state: [N, recurrent_dim]
        """
        N = grad.numel()
        g = grad.reshape(-1, 1).float()
        s = sharpness.reshape(-1, 1).float()
        inp = torch.cat([g, s], dim=1)  # [N, 2]

        # 1. Recurrent update
        h = recurrent_state.float()
        h_new = torch.tanh(self.W_h(h) + self.W_x(inp))

        # 2. ISAB: elements <-> inducing points
        x = self.input_proj(inp)
        I = self.inducing_points.unsqueeze(0)
        scale = 1.0 / math.sqrt(self.d_model)

        # Stage A: inducing points attend to elements (N -> M)
        iq = self.induce_q(I)
        ik = self.induce_k(x).unsqueeze(0)
        iv = self.induce_v(x).unsqueeze(0)
        attn = torch.softmax(torch.bmm(iq, ik.transpose(1, 2)) * scale, dim=-1)
        I_up = torch.bmm(attn, iv)

        # Stage B: elements attend to updated inducing points (M -> N)
        rq = self.read_q(x).unsqueeze(0)
        read_attn = torch.softmax(torch.bmm(rq, I_up.transpose(1, 2)) * scale, dim=-1)
        context = torch.bmm(read_attn, I_up).squeeze(0)

        # 3. PEER routing
        peer_input = torch.cat([h_new, context, inp], dim=1)
        query = self.peer_query(peer_input)
        q_a, q_b = query[:, :self.d_model // 2], query[:, self.d_model // 2:]
        idx_a = (q_a @ self.product_keys_A.T).argmax(dim=-1)
        idx_b = (q_b @ self.product_keys_B.T).argmax(dim=-1)
        expert_idx = idx_a * self.pk_dim + idx_b

        # 4. Expert evaluation
        W1 = self.expert_W1[expert_idx]
        b1 = self.expert_b1[expert_idx]
        W2 = self.expert_W2[expert_idx]
        b2 = self.expert_b2[expert_idx]
        z = torch.relu(torch.bmm(W1, g.unsqueeze(-1)).squeeze(-1) + b1)
        out = torch.bmm(W2, z.unsqueeze(-1)).squeeze(-1) + b2

        # 5. Skip connection
        smart_grad = (g + self.rescale * out).reshape(grad.shape)
        return smart_grad.to(grad.dtype), h_new

    def forward_for_bilevel(self, grad, sharpness, recurrent_state):
        """Same as forward but uses softmax routing instead of argmax for gradient flow."""
        N = grad.numel()
        g = grad.reshape(-1, 1).float()
        s = sharpness.reshape(-1, 1).float()
        inp = torch.cat([g, s], dim=1)

        h = recurrent_state.float()
        h_new = torch.tanh(self.W_h(h) + self.W_x(inp))

        x = self.input_proj(inp)
        I = self.inducing_points.unsqueeze(0)
        scale = 1.0 / math.sqrt(self.d_model)
        iq = self.induce_q(I)
        ik = self.induce_k(x).unsqueeze(0)
        iv = self.induce_v(x).unsqueeze(0)
        attn = torch.softmax(torch.bmm(iq, ik.transpose(1, 2)) * scale, dim=-1)
        I_up = torch.bmm(attn, iv)
        rq = self.read_q(x).unsqueeze(0)
        read_attn = torch.softmax(torch.bmm(rq, I_up.transpose(1, 2)) * scale, dim=-1)
        context = torch.bmm(read_attn, I_up).squeeze(0)

        # PEER with top-k sparse soft routing (differentiable, memory-efficient)
        peer_input = torch.cat([h_new, context, inp], dim=1)
        query = self.peer_query(peer_input)
        q_a, q_b = query[:, :self.d_model // 2], query[:, self.d_model // 2:]
        logits_a = q_a @ self.product_keys_A.T * 10.0  # [N, pk_dim]
        logits_b = q_b @ self.product_keys_B.T * 10.0  # [N, pk_dim]

        # Top-4 per sub-key → 16 active experts per element (instead of pk_dim^2)
        top_k = min(4, self.pk_dim)
        topk_a_vals, topk_a_idx = logits_a.topk(top_k, dim=-1)  # [N, top_k]
        topk_b_vals, topk_b_idx = logits_b.topk(top_k, dim=-1)  # [N, top_k]
        scores_a = torch.softmax(topk_a_vals, dim=-1)  # [N, top_k]
        scores_b = torch.softmax(topk_b_vals, dim=-1)  # [N, top_k]

        # Sparse outer product: [N, top_k * top_k] active experts
        # Compute expert indices from product of top-k sub-keys
        idx_a_exp = topk_a_idx.unsqueeze(2).expand(-1, -1, top_k)  # [N, top_k, top_k]
        idx_b_exp = topk_b_idx.unsqueeze(1).expand(-1, top_k, -1)  # [N, top_k, top_k]
        expert_indices = (idx_a_exp * self.pk_dim + idx_b_exp).reshape(N, -1)  # [N, top_k^2]
        routing_weights = (scores_a.unsqueeze(2) * scores_b.unsqueeze(1)).reshape(N, -1)  # [N, top_k^2]

        # Evaluate only active experts
        flat_idx = expert_indices.reshape(-1)  # [N * top_k^2]
        W1_sel = self.expert_W1[flat_idx]  # [N*top_k^2, expert_hidden, 1]
        b1_sel = self.expert_b1[flat_idx]  # [N*top_k^2, expert_hidden]
        W2_sel = self.expert_W2[flat_idx]  # [N*top_k^2, 1, expert_hidden]
        b2_sel = self.expert_b2[flat_idx]  # [N*top_k^2, 1]
        g_rep = g.repeat_interleave(top_k * top_k, dim=0)  # [N*top_k^2, 1]
        z = torch.relu(torch.bmm(W1_sel, g_rep.unsqueeze(-1)).squeeze(-1) + b1_sel)
        expert_out = torch.bmm(W2_sel, z.unsqueeze(-1)).squeeze(-1) + b2_sel  # [N*top_k^2, 1]
        expert_out = expert_out.reshape(N, top_k * top_k)  # [N, top_k^2]
        out = (routing_weights * expert_out).sum(dim=-1, keepdim=True)  # [N, 1]

        smart_grad = (g + self.rescale * out).reshape(grad.shape)
        return smart_grad.to(grad.dtype), h_new

    def get_weights(self):
        """Extract weights for CUDA kernel."""
        return {
            'inducing_points': self.inducing_points.detach().float().contiguous(),
            'input_proj_W': self.input_proj.weight.detach().float().contiguous(),
            'input_proj_b': self.input_proj.bias.detach().float().contiguous(),
            'induce_q_W': self.induce_q.weight.detach().float().contiguous(),
            'induce_k_W': self.induce_k.weight.detach().float().contiguous(),
            'induce_v_W': self.induce_v.weight.detach().float().contiguous(),
            'read_q_W': self.read_q.weight.detach().float().contiguous(),
            'W_h': self.W_h.weight.detach().float().contiguous(),
            'W_x_W': self.W_x.weight.detach().float().contiguous(),
            'W_x_b': self.W_x.bias.detach().float().contiguous(),
            'peer_query_W': self.peer_query.weight.detach().float().contiguous(),
            'product_keys_A': self.product_keys_A.detach().float().contiguous(),
            'product_keys_B': self.product_keys_B.detach().float().contiguous(),
            'expert_W1': self.expert_W1.detach().float().contiguous(),
            'expert_b1': self.expert_b1.detach().float().contiguous(),
            'expert_W2': self.expert_W2.detach().float().contiguous(),
            'expert_b2': self.expert_b2.detach().float().contiguous(),
            'rescale': self.rescale,
            'num_inducing': self.num_inducing,
            'd_model': self.d_model,
            'pk_dim': self.pk_dim,
            'expert_hidden': self.expert_hidden,
            'recurrent_dim': self.recurrent_dim,
        }
