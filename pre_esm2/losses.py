import torch
import torch.nn as nn

from .config import TEMPERATURE, DEVICE


class SupConLoss(nn.Module):
    """监督对比学习损失"""

    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # 归一化特征向量
        features = nn.functional.normalize(features, p=2, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 创建同类掩码
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(DEVICE)

        # 排除自身对角线
        self_mask = torch.eye(labels.size(0), dtype=torch.bool).to(DEVICE)
        mask = mask.masked_fill(self_mask, 0)

        # 计算 logits
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(
            exp_sim.sum(dim=1, keepdim=True) + 1e-8
        )  # 添加小常数避免 log(0)

        # 计算每个样本的平均正样本对数概率
        mean_log_prob = (mask * log_prob).sum(dim=1) / (
            mask.sum(dim=1) + 1e-8
        )  # 添加小常数避免除以 0

        # 最终的对比损失
        loss = -mean_log_prob.mean()
        return loss