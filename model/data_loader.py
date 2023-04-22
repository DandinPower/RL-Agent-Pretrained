import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
load_dotenv()

TRACE_PATH = os.getenv('TRACE_PATH')
TRACE_LENGTH = int(os.getenv('TRACE_LENGTH'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
TRAIN_TEST_SPLIT = float(os.getenv('TRAIN_TEST_SPLIT'))

class MSR_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx, [2, 3]]
        y = self.data[idx, 4]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def GetDataLoaders():
    data = pd.read_csv(TRACE_PATH, header=None, dtype=np.float64).values

    # Transfer col lba into lba_diff
    data[1:, 2] = np.diff(data[:, 2])
    data[0, 2] = 0

    # Standardize x data (lba_diff)
    scaler_diff = StandardScaler()
    data[:, 2] = scaler_diff.fit_transform(data[:, 2].reshape(-1, 1)).flatten()

    # Standardize x data (bytes)
    scaler_bytes = StandardScaler()
    data[:, 3] = scaler_bytes.fit_transform(data[:, 3].reshape(-1, 1)).flatten()

    # Split the data into train and test sets
    train_data, test_data = train_test_split(data, test_size=TRAIN_TEST_SPLIT, shuffle=False, random_state=None)

    # Create training dataset and DataLoader
    train_dataset = MSR_Dataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create test dataset and DataLoader
    test_dataset = MSR_Dataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, test_dataloader
