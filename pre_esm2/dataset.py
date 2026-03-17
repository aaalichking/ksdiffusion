import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    POSITIVE_CSV,
    NEGATIVE_CSV,
    MLM_PROBABILITY,
    MAX_LENGTH,
    BATCH_SIZE,
)


class PeptideDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length: int = MAX_LENGTH):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # 编码序列
        encoding = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 创建 MLM 掩码
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # 创建 MLM 标签（原始输入 ID）
        mlm_labels = input_ids.clone()

        # 随机选择一定比例 token 进行掩码
        probability_matrix = torch.full(mlm_labels.shape, MLM_PROBABILITY)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% 的时间用 [MASK] 替换
        indices_replaced = (
            torch.bernoulli(torch.full(mlm_labels.shape, 0.8)).bool()
            & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% 用随机 token，10% 保持原 token
        indices_random = (
            torch.bernoulli(torch.full(mlm_labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), mlm_labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        # 未被掩码的位置标签设为 -100（忽略）
        mlm_labels[~masked_indices] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mlm_labels": mlm_labels,
            "scl_labels": torch.tensor(label, dtype=torch.long),
        }


def create_dataloaders(tokenizer):
    """创建训练和验证 DataLoader"""
    # 加载数据
    pos_df = pd.read_csv(POSITIVE_CSV)
    neg_df = pd.read_csv(NEGATIVE_CSV)

    # 添加标签
    pos_df["label"] = 1
    neg_df["label"] = 0

    # 合并数据集
    full_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # 划分训练集和验证集
    train_df, val_df = train_test_split(
        full_df,
        test_size=0.2,
        random_state=42,
        stratify=full_df["label"],
    )

    # 创建数据集
    train_dataset = PeptideDataset(
        train_df["PEPTIDE"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        MAX_LENGTH,
    )

    val_dataset = PeptideDataset(
        val_df["PEPTIDE"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
        MAX_LENGTH,
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader