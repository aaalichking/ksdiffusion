import math
import torch
import torch.nn as nn


class DiffusionProcess:
    """
    扩散过程工具类：生成 beta 调度、添加噪声、计算损失权重等
    """
    def __init__(self, steps: int = 500, device: str = 'cpu'):
        self.steps = steps
        self.device = device
        self.betas = self.cosine_beta_schedule(steps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def cosine_beta_schedule(self, steps: int, s: float = 0.008) -> torch.Tensor:
        """余弦调度函数"""
        x = torch.linspace(0, steps, steps + 1, device=self.device)
        alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        向输入添加噪声
        x_0: [B, L, D]
        t: [B]
        """
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        return x_t, noise

    def get_loss_weights(self, t: torch.Tensor):
        """获取不同时间步的损失权重，目前使用均匀权重"""
        return 1.0