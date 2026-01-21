"""Mixture of Experts (MoE) modules for emotion recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single expert network (FFN)."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k routing.

    Architecture:
        Input → Router → Top-k Experts → Weighted Sum → Output

    Args:
        d_model: Model dimension
        d_ffn: FFN hidden dimension (default: 4 * d_model)
        num_experts: Number of expert networks
        top_k: Number of experts to route to per token (default: 2)
        dropout: Dropout rate
        load_balance_weight: Weight for load balancing loss (default: 0.01)
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int | None = None,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn or d_model * 4
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.load_balance_weight = load_balance_weight

        # Router: 线性层将 token 映射到 expert 分数
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Expert networks
        self.experts = nn.ModuleList([
            Expert(d_model, self.d_ffn, dropout) for _ in range(num_experts)
        ])

        # 记录 load balancing loss (用于训练)
        self.aux_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, D]

        Returns:
            Output tensor [B, T, D]
        """
        B, T, D = x.shape

        # Router logits: [B, T, num_experts]
        router_logits = self.router(x)

        # Top-k routing
        router_probs = F.softmax(router_logits, dim=-1)  # [B, T, num_experts]
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize top-k probs
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute load balancing loss (auxiliary loss)
        if self.training:
            self._compute_load_balance_loss(router_probs)

        # Dispatch to experts and combine
        # 简化实现：遍历 experts（对于少量 experts 足够高效）
        output = torch.zeros_like(x)

        for k in range(self.top_k):
            expert_indices = top_k_indices[:, :, k]  # [B, T]
            expert_probs = top_k_probs[:, :, k]  # [B, T]

            for e in range(self.num_experts):
                # 找到被路由到 expert e 的 tokens
                mask = (expert_indices == e)  # [B, T]
                if mask.any():
                    # 收集这些 tokens
                    expert_input = x[mask]  # [N, D]
                    expert_output = self.experts[e](expert_input)  # [N, D]

                    # 加权累加到输出
                    weighted_output = expert_output * expert_probs[mask].unsqueeze(-1)
                    output[mask] += weighted_output

        return output

    def _compute_load_balance_loss(self, router_probs: torch.Tensor):
        """Compute auxiliary load balancing loss to encourage balanced expert usage.

        Based on Switch Transformer paper.
        """
        # router_probs: [B, T, num_experts]
        # 计算每个 expert 被选中的比例
        expert_usage = router_probs.mean(dim=[0, 1])  # [num_experts]

        # 理想情况：每个 expert 使用 1/num_experts
        ideal_usage = 1.0 / self.num_experts

        # Load balance loss: 鼓励均匀分布
        # L_aux = num_experts * sum(f_i * P_i)
        # f_i = fraction of tokens routed to expert i
        # P_i = average router probability for expert i
        tokens_per_expert = router_probs.sum(dim=[0, 1])  # [num_experts]
        total_tokens = router_probs.shape[0] * router_probs.shape[1]
        fraction = tokens_per_expert / total_tokens

        self.aux_loss = self.load_balance_weight * self.num_experts * (fraction * expert_usage).sum()


class MoEMambaUpdater(nn.Module):
    """Mamba-based updater with MoE FFN layers.

    Architecture per layer:
        Input → LayerNorm → Mamba → Dropout → Residual →
        LayerNorm → MoE-FFN → Dropout → Residual → Output

    Args:
        d_input: Input dimension
        d_model: Model dimension for Mamba blocks
        d_output: Output dimension
        n_layers: Number of Mamba + MoE layers
        num_experts: Number of experts per MoE layer
        top_k: Number of experts to route to
        d_state: SSM state expansion factor
        d_conv: Local convolution width
        expand: Block expansion factor
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_input: int = 1088,
        d_model: int = 256,
        d_output: int = 64,
        n_layers: int = 2,
        num_experts: int = 4,
        top_k: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        try:
            from mamba_ssm import Mamba
        except ImportError:
            raise ImportError("mamba-ssm is required. Install with: pip install mamba-ssm")

        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)

        # Mamba blocks
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])

        # MoE FFN layers (after each Mamba)
        self.moe_layers = nn.ModuleList([
            MoELayer(
                d_model=d_model,
                d_ffn=d_model * 4,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                load_balance_weight=load_balance_weight,
            )
            for _ in range(n_layers)
        ])

        # Layer norms (pre-norm for Mamba, pre-norm for MoE)
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.moe_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # Dropouts
        self.mamba_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.moe_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_output),
            nn.LayerNorm(d_output),
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: Input [B, T, d_input]

        Returns:
            Output [B, T, d_output]
        """
        x = self.input_proj(z_t)  # [B, T, d_model]

        # Handle 2D input
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        for i in range(self.n_layers):
            # Mamba sub-layer
            residual = x
            x = self.mamba_norms[i](x)
            x = self.mamba_layers[i](x)
            x = self.mamba_dropouts[i](x)
            x = x + residual

            # MoE FFN sub-layer
            residual = x
            x = self.moe_norms[i](x)
            x = self.moe_layers[i](x)
            x = self.moe_dropouts[i](x)
            x = x + residual

        if is_2d:
            x = x.squeeze(1)

        return self.output_proj(x)

    def get_aux_loss(self) -> torch.Tensor:
        """Get total auxiliary loss from all MoE layers."""
        total_loss = 0.0
        for moe in self.moe_layers:
            total_loss = total_loss + moe.aux_loss
        return total_loss
