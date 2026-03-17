from torch.utils.data import Dataset
import pandas as pd


class ProteinSequenceDataset(Dataset):
    """
    只使用目标数据集的简单 Dataset
    读取 CSV 中的 'PEPTIDE' 列作为序列
    """
    def __init__(self, data_csv: str, max_length: int = 61):
        df = pd.read_csv(data_csv)
        self.sequences = df['PEPTIDE'].tolist()
        print(f"Loaded {len(self.sequences)} peptide sequences")
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]