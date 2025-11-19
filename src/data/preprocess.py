import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import StandardScaler

class UnivariateTimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len=7, target_col="close", normalize=False):
        self.data = data.dropna().reset_index(drop=True)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col
        self.normalize = normalize
        self.series = self.data[target_col].values
        
        if normalize:
            self.mean = np.mean(self.series)
            self.std = np.std(self.series)
    
    def __len__(self):
        return len(self.series) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        x = self.series[idx:idx+self.seq_len].copy()
        y = self.series[idx+self.seq_len:idx+self.seq_len+self.pred_len].copy()
        
        if self.normalize:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std
        
        return torch.FloatTensor(x), torch.FloatTensor(y)


class NormalizedDataset(Dataset):
    def __init__(self, original_dataset, scaler=None):
        self.original_dataset = original_dataset
        
        if scaler is None:
            all_data = []
            for i in range(len(original_dataset)):
                x, y = original_dataset[i]
                all_data.extend(x.numpy())
                all_data.extend(y.numpy())
            
            self.scaler = StandardScaler()
            self.scaler.fit(np.array(all_data).reshape(-1, 1))
        else:
            self.scaler = scaler
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        x_norm = self.scaler.transform(x.numpy().reshape(-1, 1)).flatten()
        y_norm = self.scaler.transform(y.numpy().reshape(-1, 1)).flatten()
        return torch.FloatTensor(x_norm), torch.FloatTensor(y_norm)


def create_univariate_datasets(df, seq_lengths, pred_len=7, target_col="close"):
    """Create univariate datasets for different sequence lengths"""
    datasets = {}

    for seq_len in seq_lengths:
        dataset = UnivariateTimeSeriesDataset(
        data=df, seq_len=seq_len, pred_len=pred_len,
        target_col=target_col, normalize=False
        )
        datasets[f"{seq_len}d"] = dataset

    return datasets


def create_time_based_splits(dataset, train_ratio=0.7, val_ratio=0.15):
    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    
    train_indices = list(range(0, train_len))
    val_indices = list(range(train_len, train_len + val_len))
    test_indices = list(range(train_len + val_len, total_len))
    
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices)
    )
