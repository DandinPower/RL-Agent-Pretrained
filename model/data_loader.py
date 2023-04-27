import pandas as pd
import numpy as np
import torch
import csv
import statistics
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
LBA_FREQ_PATH = os.getenv('LBA_FREQ_PATH')

# Define a function to standardize the frequency values
def standardize(frequency_values):
    mean = statistics.mean(frequency_values)
    stdev = statistics.stdev(frequency_values)
    standardized_values = [(x - mean) / stdev for x in frequency_values]
    return standardized_values

def GetLbaFreqDict():
    # Load the LBA frequency statistics from the CSV file
    lba_frequencies = {}
    with open(LBA_FREQ_PATH, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lba = row['LBA']
            frequency = int(row['Frequency'])
            lba_frequencies[lba] = frequency

    # Standardize the frequency values
    standardized_frequencies = standardize(list(lba_frequencies.values()))

    # Store the standardized frequency values in a dictionary with the LBA values as keys
    lba_standardized_frequencies = {}
    for i, lba in enumerate(lba_frequencies):
        frequency = standardized_frequencies[i]
        lba_standardized_frequencies[lba] = frequency
    return lba_standardized_frequencies

class MSR_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx, [1, 2, 3]]
        y = self.data[idx, 4]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def GetDataLoaders():
    data = pd.read_csv(TRACE_PATH, header=None, dtype=np.float64).values

    # copy lba into col 1
    lbaFreqDict = GetLbaFreqDict()
    for i in range(len(data)):
        index = str(int(data[i, 2]))
        if index in lbaFreqDict:
            data[i, 1] = lbaFreqDict[index]

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
