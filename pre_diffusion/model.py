import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * (-embeddings))
        embeddings = t.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x, scale=None, shift=None):
        # 自适应归一化
        if scale is not None and shift is not None:
            x_norm = self.norm1(x) * (1 + scale) + shift
        else:
            x_norm = self.norm1(x)

        # 注意力
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x


class ProteinDiT(nn.Module):
    def __init__(self, input_dim: int = 1280, hidden_dim: int = 512, depth: int = 8, heads: int = 8):
        super().__init__()
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 时间步嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # DiT 块
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, heads, mlp_ratio=4) for _ in range(depth)
        ])

        # 自适应归一化
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)
        )

        # 输出层
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] (L=61, D=1280)
        h = self.input_proj(x)  # [B, L, H]

        # 时间嵌入
        t_emb = self.time_embed(t)  # [B, H]

        # 自适应调制参数
        mod_params = self.adaLN_modulation(t_emb)
        scale1, shift1, scale2, shift2, gate_scale, gate_shift = mod_params.chunk(6, dim=1)

        # DiT 块（目前只使用 scale1, shift1）
        for block in self.blocks:
            h = block(h, scale=scale1.unsqueeze(1), shift=shift1.unsqueeze(1))

        # 门控输出
        h = self.output_norm(h) * (1 + gate_scale.unsqueeze(1)) + gate_shift.unsqueeze(1)
        return self.output_proj(h)