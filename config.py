import os
import random
import tempfile
import numpy as np
import torch


class Config:
    def __init__(self, rare_kinase_csv):
        # 数据集路径
        self.rare_kinase_csv = rare_kinase_csv
        self.kinase_mid_dir = "kinase_low"  # 正样本目录
        self.kinase_group_dir = "kinase_group_all"  # 困难负样本目录
        self.pretrain_dir = "pretrain"  # 简单负样本目录

        # 模型参数 - 更新路径
        self.device = torch.device(
            "cuda:1" if torch.cuda.is_available() else "cpu")
        self.esm_model_path = "local_esm_model"  # ESM2模型路径
        self.diffusion_model_path = "diffusion_model_checkpoints/best_model.pth"  # 扩散模型路径
        self.max_length = 61
        self.embed_dim = 1280
        self.hidden_dim = 512

        # 数据增强参数 - 使用10%, 20%, 30%增强倍数
        self.augmentation_factors = [0.1, 0.2, 0.3]  # 增强倍数
        self.negative_ratio = 1.0  # 负样本比例（相对于正样本）
        self.generation_factor = 10  # 生成样本是目标样本的倍数（10倍）
        self.min_augmentation_factor = 0.1  # 最小增强比例
        self.max_regeneration_attempts = 5  # 最大重新生成次数

        # 扩散模型参数
        self.diffusion_steps = 500
        self.generation_steps = 100

        # 分类器参数 - 添加正则化
        self.classifier_lr = 1e-4
        self.classifier_epochs = 200
        self.classifier_batch_size = 32
        self.freeze_esm = True  # 是否冻结ESM参数
        self.weight_decay = 1e-4  # L2正则化系数

        # 微调参数
        self.meta_learning_steps = 1000
        self.adv_finetune_epochs = 500
        self.adv_batch_size = 4

        # 输出目录 - 改为result_low6
        self.output_dir = os.path.join(
            "result_low6",
            os.path.basename(self.rare_kinase_csv).replace(".csv", "_result")
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # 临时文件目录
        self.temp_dir = tempfile.mkdtemp(prefix="kinase_temp_")
        print(f"Using temporary directory: {self.temp_dir}")

        # 设置随机种子
        self.seed = 3407
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)