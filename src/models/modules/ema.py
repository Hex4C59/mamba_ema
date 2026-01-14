import torch
import torch.nn as nn


class EMAState(nn.Module):
    def __init__(
        self,
        d_state: int = 64,
        alpha: float = 0.8,
        learnable_alpha: bool = False,
        separate_va: bool = False,
    ) -> None:
        super().__init__()
        self.d_state = d_state
        self.separate_va = separate_va

        if learnable_alpha:
            if separate_va:
                # Separate alpha for V and A (often A changes faster)
                self.alpha_v = nn.Parameter(torch.tensor(alpha))
                self.alpha_a = nn.Parameter(torch.tensor(alpha - 0.1))
            else:
                self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            if separate_va:
                self.register_buffer("alpha_v", torch.tensor(alpha))
                self.register_buffer("alpha_a", torch.tensor(alpha - 0.1))
            else:
                self.register_buffer("alpha", torch.tensor(alpha))

        self.learnable_alpha = learnable_alpha

    def forward(self, u_t: torch.Tensor, c_prev: torch.Tensor | None = None) -> torch.Tensor:

        # Initialize state if needed
        if c_prev is None:
            c_prev = torch.zeros_like(u_t)

        if self.separate_va:
            d_half = self.d_state // 2
            u_v, u_a = u_t[:, :d_half], u_t[:, d_half:]
            c_v, c_a = c_prev[:, :d_half], c_prev[:, d_half:]

            alpha_v = self.alpha_v
            alpha_a = self.alpha_a

            if self.learnable_alpha:
                alpha_v = torch.clamp(alpha_v, 0.6, 0.9)
                alpha_a = torch.clamp(alpha_a, 0.6, 0.9)

            c_v_new = alpha_v * c_v + (1 - alpha_v) * u_v
            c_a_new = alpha_a * c_a + (1 - alpha_a) * u_a

            c_t = torch.cat([c_v_new, c_a_new], dim=-1)
        else:
            alpha = self.alpha
            if self.learnable_alpha:
                alpha = torch.clamp(alpha, 0.6, 0.9)

            c_t = alpha * c_prev + (1 - alpha) * u_t

        return c_t

    def reset_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.d_state, device=device)
