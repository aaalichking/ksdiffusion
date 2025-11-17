import os
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from transformers import EsmTokenizer


class KinaseDataProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = EsmTokenizer.from_pretrained(config.esm_model_path)

    def load_and_split_rare_kinase(self):
        """加载并划分稀少激酶数据集 - 改进数据分割"""
        df = pd.read_csv(self.config.rare_kinase_csv)
        total_samples = len(df)

        # 设置测试集最小数量
        min_test_size = 50

        # 根据样本量动态调整划分比例
        if total_samples > 5 * min_test_size:
            test_size = max(min_test_size, total_samples // 6)
            train_size = total_samples - test_size
        else:
            test_size = min(min_test_size, total_samples)
            train_size = total_samples - test_size

        print(f"Total positive samples: {total_samples}")
        print(f"Train positive samples: {train_size}")
        print(f"Test positive samples: {test_size}")

        # 划分训练集和测试集 - 严格分层
        stratify_col = df['KINASE_FAMILY'] if 'KINASE_FAMILY' in df.columns else None
        train_df, test_df = train_test_split(
            df,
            train_size=train_size,
            test_size=test_size,
            random_state=self.config.seed,
            stratify=stratify_col
        )

        # 确保训练集和测试集没有重叠序列
        train_peptides = set(train_df['PEPTIDE'])
        test_df = test_df[~test_df['PEPTIDE'].isin(train_peptides)]

        # 从训练集中再划分验证集 (20%)
        val_size = int(0.2 * len(train_df))
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_size,
            random_state=self.config.seed,
            stratify=train_df['KINASE_FAMILY'] if 'KINASE_FAMILY' in train_df.columns else None
        )

        print(f"Final sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df

    def load_negative_samples(self, exclude_peptides=None):
        """加载负样本数据集"""
        hard_negatives = []
        current_file = os.path.basename(self.config.rare_kinase_csv)

        # 从kinase_group_all加载除当前文件外的所有文件作为困难负样本
        for file in os.listdir(self.config.kinase_group_dir):
            file_path = os.path.join(self.config.kinase_group_dir, file)
            if not os.path.isfile(file_path) or not file.endswith('.csv'):
                continue
            if file == current_file:
                continue
            try:
                df = pd.read_csv(file_path)
                hard_negatives.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue

        hard_negatives = pd.concat(hard_negatives, ignore_index=True) if hard_negatives else pd.DataFrame()

        # 加载简单负样本
        easy_negatives = []
        for file in os.listdir(self.config.pretrain_dir):
            file_path = os.path.join(self.config.pretrain_dir, file)
            if not os.path.isfile(file_path) or not file.endswith(".csv"):
                continue
            try:
                df = pd.read_csv(file_path)
                if 'label' in df.columns:
                    df = df[df['label'] == 0]
                easy_negatives.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue

        easy_negatives = pd.concat(easy_negatives, ignore_index=True) if easy_negatives else pd.DataFrame()

        if exclude_peptides is not None:
            if not hard_negatives.empty and 'PEPTIDE' in hard_negatives.columns:
                hard_negatives = hard_negatives[~hard_negatives['PEPTIDE'].isin(exclude_peptides)]
            if not easy_negatives.empty and 'PEPTIDE' in easy_negatives.columns:
                easy_negatives = easy_negatives[~easy_negatives['PEPTIDE'].isin(exclude_peptides)]

        return hard_negatives, easy_negatives

    def prepare_datasets(self):
        """准备完整的数据集 - 包含验证集"""
        # 划分稀少激酶数据集
        train_pos, val_pos, test_pos = self.load_and_split_rare_kinase()

        # 获取所有样本的肽序列用于排除
        all_peptides = set(train_pos['PEPTIDE']) | set(val_pos['PEPTIDE']) | set(test_pos['PEPTIDE'])

        # 加载负样本，排除正样本肽序列
        train_hard_neg, train_easy_neg = self.load_negative_samples(exclude_peptides=all_peptides)
        val_hard_neg, val_easy_neg = self.load_negative_samples(exclude_peptides=all_peptides)
        test_hard_neg, test_easy_neg = self.load_negative_samples(exclude_peptides=all_peptides)

        # ===== 训练集负样本采样 =====
        num_train_neg = int(self.config.negative_ratio * len(train_pos))
        num_hard_neg = min(num_train_neg // 2, len(train_hard_neg))
        num_easy_neg = min(num_train_neg - num_hard_neg, len(train_easy_neg))

        if not train_hard_neg.empty and num_hard_neg > 0:
            train_hard_neg = train_hard_neg.sample(num_hard_neg, random_state=self.config.seed)
        else:
            train_hard_neg = pd.DataFrame()

        if not train_easy_neg.empty and num_easy_neg > 0:
            train_easy_neg = train_easy_neg.sample(num_easy_neg, random_state=self.config.seed)
        else:
            train_easy_neg = pd.DataFrame()

        train_neg = pd.concat([train_hard_neg, train_easy_neg])
        if not train_neg.empty:
            train_neg['label'] = 0.0
        else:
            train_neg = pd.DataFrame()

        # ===== 验证集负样本采样 =====
        num_val_neg = len(val_pos)
        num_hard_neg = min(num_val_neg // 2, len(val_hard_neg))
        num_easy_neg = min(num_val_neg - num_hard_neg, len(val_easy_neg))

        if not val_hard_neg.empty and num_hard_neg > 0:
            val_hard_neg = val_hard_neg.sample(num_hard_neg, random_state=self.config.seed)
        else:
            val_hard_neg = pd.DataFrame()

        if not val_easy_neg.empty and num_easy_neg > 0:
            val_easy_neg = val_easy_neg.sample(num_easy_neg, random_state=self.config.seed)
        else:
            val_easy_neg = pd.DataFrame()

        val_neg = pd.concat([val_hard_neg, val_easy_neg])
        if not val_neg.empty:
            val_neg['label'] = 0.0
        else:
            val_neg = pd.DataFrame()

        # ===== 测试集负样本采样 =====
        num_test_neg = len(test_pos)
        num_hard_neg = min(num_test_neg // 2, len(test_hard_neg))
        num_easy_neg = min(num_test_neg - num_hard_neg, len(test_easy_neg))

        if not test_hard_neg.empty and num_hard_neg > 0:
            test_hard_neg = test_hard_neg.sample(num_hard_neg, random_state=self.config.seed)
        else:
            test_hard_neg = pd.DataFrame()

        if not test_easy_neg.empty and num_easy_neg > 0:
            test_easy_neg = test_easy_neg.sample(num_easy_neg, random_state=self.config.seed)
        else:
            test_easy_neg = pd.DataFrame()

        test_neg = pd.concat([test_hard_neg, test_easy_neg])
        if not test_neg.empty:
            test_neg['label'] = 0.0
        else:
            test_neg = pd.DataFrame()

        # 准备正样本
        train_pos['label'] = 1.0
        val_pos['label'] = 1.0
        test_pos['label'] = 1.0

        # 合并数据集
        train_df = pd.concat([train_pos, train_neg]) if not train_neg.empty else train_pos
        val_df = pd.concat([val_pos, val_neg]) if not val_neg.empty else val_pos
        test_df = pd.concat([test_pos, test_neg]) if not test_neg.empty else test_pos

        # 确保所有标签都是数值类型
        train_df['label'] = pd.to_numeric(train_df['label'], errors='coerce').fillna(0).astype(float)
        val_df['label'] = pd.to_numeric(val_df['label'], errors='coerce').fillna(0).astype(float)
        test_df['label'] = pd.to_numeric(test_df['label'], errors='coerce').fillna(0).astype(float)

        print(f"Final train size: {len(train_df)} (Pos: {len(train_pos)}, Neg: {len(train_neg)})")
        print(f"Final val size: {len(val_df)} (Pos: {len(val_pos)}, Neg: {len(val_neg)})")
        print(f"Final test size: {len(test_df)} (Pos: {len(test_pos)}, Neg: {len(test_neg)})")

        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

    def resample_negatives(self, train_df, target_neg_count):
        """在数据增强后重新采样负样本以平衡数据集，确保1:1比例"""
        existing_peptides = set(train_df['PEPTIDE'])
        hard_neg, easy_neg = self.load_negative_samples(exclude_peptides=existing_peptides)

        current_neg_count = len(train_df[train_df['label'] == 0])
        needed_neg_count = max(0, target_neg_count - current_neg_count)

        if needed_neg_count <= 0:
            print("No additional negatives needed. Dataset is balanced.")
            return train_df

        hard_samples = pd.DataFrame()
        easy_samples = pd.DataFrame()

        num_hard_needed = needed_neg_count // 2
        num_easy_needed = needed_neg_count - num_hard_needed

        if not hard_neg.empty and num_hard_needed > 0:
            num_hard_available = min(num_hard_needed, len(hard_neg))
            if num_hard_available > 0:
                hard_samples = hard_neg.sample(n=num_hard_available, random_state=self.config.seed)
                hard_samples['label'] = 0.0
                print(f"Added {num_hard_available} hard negative samples")

        if not easy_neg.empty and num_easy_needed > 0:
            num_easy_available = min(num_easy_needed, len(easy_neg))
            if num_easy_available > 0:
                easy_samples = easy_neg.sample(n=num_easy_available, random_state=self.config.seed)
                easy_samples['label'] = 0.0
                print(f"Added {num_easy_available} easy negative samples")

        new_neg_samples = pd.concat([hard_samples, easy_samples], ignore_index=True)
        balanced_train = pd.concat([train_df, new_neg_samples], ignore_index=True)

        balanced_train['label'] = pd.to_numeric(
            balanced_train['label'], errors='coerce'
        ).fillna(0).astype(float)

        pos_count = len(balanced_train[balanced_train['label'] == 1])
        neg_count = len(balanced_train[balanced_train['label'] == 0])
        print(f"Balanced dataset: {pos_count} positive, {neg_count} negative samples")

        return balanced_train