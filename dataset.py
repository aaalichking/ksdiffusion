import torch
from torch.utils.data import Dataset
import pandas as pd


class KinaseDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.sequences = df['PEPTIDE'].tolist()
        self.labels = pd.to_numeric(df['label'], errors='coerce').fillna(0).values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }