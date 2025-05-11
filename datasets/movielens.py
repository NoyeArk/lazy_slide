import torch
from torch.utils.data import Dataset
import pandas as pd


class MovieDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data = self.data.fillna(0)
        self.data = self.data.apply(pd.to_numeric, errors='coerce')
        self.labels = self.data['label'].values
        self.data = self.data.drop(columns=['label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        label = self.labels[idx]
        return torch.tensor(sample.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
