import torch
import os

# 配置参数 - 直接使用本地模型路径
MODEL_NAME = "local_esm_model"  # 本地模型目录名称
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_NAME)

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "training_checkpoint.pth")
POSITIVE_CSV = os.path.join(os.path.dirname(__file__), "formatted_peptides.csv")
NEGATIVE_CSV = os.path.join(os.path.dirname(__file__), "negative_peptides.csv")

# 训练参数
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-5
TEMPERATURE = 0.1        # 对比学习温度参数
MLM_PROBABILITY = 0.15   # MLM 掩码概率
MAX_LENGTH = 128         # 最大序列长度
SCL_WEIGHT = 0.5         # 监督对比学习损失权重
MLM_WEIGHT = 0.5         # MLM 损失权重

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")