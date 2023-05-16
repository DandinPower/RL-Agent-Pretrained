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


BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
TRAIN_TEST_SPLIT = float(os.getenv('TRAIN_TEST_SPLIT'))

TRACE_PATH = os.getenv('TRACE_PATH')
TRACE_LENGTH = int(os.getenv('TRACE_LENGTH'))
TRACE_2_PATH = os.getenv('TRACE_2_PATH')
TRACE_2_LENGTH = int(os.getenv('TRACE_2_LENGTH'))

LBA_FREQ_PATH = os.getenv('LBA_FREQ_PATH')
LBA_FREQ_2_PATH = os.getenv('LBA_FREQ_2_PATH')


# Define a function to standardize the frequency values
def standardize(frequency_values):
    mean = statistics.mean(frequency_values)
    stdev = statistics.stdev(frequency_values)
    standardized_values = [(x - mean) / stdev for x in frequency_values]
    return standardized_values

def GetLbaFreqDict(path):
    # Load the LBA frequency statistics from the CSV file
    lba_frequencies = {}
    with open(path, newline='') as csvfile:
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
    data1 = pd.read_csv(TRACE_PATH, header=None, dtype=np.float64).values
    
    # copy lba into col 1
    lbaFreqDict = GetLbaFreqDict(LBA_FREQ_PATH)
    for i in range(len(data1)):
        index = str(int(data1[i, 2]))
        if index in lbaFreqDict:
            data1[i, 1] = lbaFreqDict[index]
    
    # Transfer col lba into lba_diff
    data1[1:, 2] = np.diff(data1[:, 2])
    data1[0, 2] = 0

    # Standardize x data (lba_diff)
    scaler_diff = StandardScaler()
    data1[:, 2] = scaler_diff.fit_transform(data1[:, 2].reshape(-1, 1)).flatten()

    # Standardize x data (bytes)
    scaler_bytes = StandardScaler()
    data1[:, 3] = scaler_bytes.fit_transform(data1[:, 3].reshape(-1, 1)).flatten()


    data2 = pd.read_csv(TRACE_2_PATH, header=None, dtype=np.float64).values

    # copy lba into col 1
    lbaFreqDict = GetLbaFreqDict(LBA_FREQ_2_PATH)
    for i in range(len(data2)):
        index = str(int(data2[i, 2]))
        if index in lbaFreqDict:
            data2[i, 1] = lbaFreqDict[index]

    # Transfer col lba into lba_diff
    data2[1:, 2] = np.diff(data2[:, 2])
    data2[0, 2] = 0

    # Standardize x data (lba_diff)
    scaler_diff = StandardScaler()
    data2[:, 2] = scaler_diff.fit_transform(data2[:, 2].reshape(-1, 1)).flatten()

    # Standardize x data (bytes)
    scaler_bytes = StandardScaler()
    data2[:, 3] = scaler_bytes.fit_transform(data2[:, 3].reshape(-1, 1)).flatten()

    data = np.concatenate((data1, data2), axis=0)

    # Split the data into train and test sets
    train_data, test_data = train_test_split(data, test_size=TRAIN_TEST_SPLIT, shuffle=True, random_state=42)

    # Create training dataset and DataLoader
    train_dataset = MSR_Dataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create test dataset and DataLoader
    test_dataset = MSR_Dataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, test_dataloader
