import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from model.q_model import QModel
from model.data_loader import GetDataLoaders
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()

EPOCH = int(os.getenv('EPOCH'))
LR = float(os.getenv('LR'))
MODEL_NAME = os.getenv('MODEL_NAME')
MODEL_WEIGHT_DIR = os.getenv('MODEL_WEIGHT_DIR')
TRAIN_PROGRESS_DIR = os.getenv('TRAIN_PROGRESS_DIR')

def Train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = QModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    train_dataloader, test_dataloader = GetDataLoaders()
    print(f'Start Training...')
    
    # Lists to store training and validation losses
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCH):
        print(f"Epoch {epoch + 1}/{EPOCH}")

        # Train on training data
        train_loss = 0.0
        num_train_batches = 0
        for batch_x, batch_y in tqdm(train_dataloader, desc="Training", unit="batch"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = Optimize(model, optimizer, loss_fn, batch_x, batch_y)
            train_loss += loss
            num_train_batches += 1
        train_avg_loss = train_loss / num_train_batches
        train_losses.append(train_avg_loss)  # Append average training loss
        
        # Validate on test data
        total_loss = 0.0
        num_batches = 0
        for batch_x, batch_y in tqdm(test_dataloader, desc="Validating", unit="batch"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = Validate(model, loss_fn, batch_x, batch_y)
            total_loss += loss
            num_batches += 1
        avg_loss = total_loss / num_batches
        val_losses.append(avg_loss)  # Append validation loss
        print(f"Train loss: {train_avg_loss:.4f}")
        print(f"Validation loss: {avg_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), f'{MODEL_WEIGHT_DIR}/{MODEL_NAME}.pth')
    print("Model saved successfully.")
        
    # Plot training and validation losses
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(f'{TRAIN_PROGRESS_DIR}/{MODEL_NAME}.png', dpi=300)
    plt.clf()

def Optimize(model, optimizer, loss_fn, batch_x, batch_y):
    model.train()
    optimizer.zero_grad()
    predictions = model(batch_x)
    loss = loss_fn(predictions, batch_y)
    loss.backward()
    optimizer.step()
    return loss.item()  # Return the loss value

def Validate(model, loss_fn, batch_x, batch_y):
    model.eval()
    with torch.no_grad():
        predictions = model(batch_x)
        loss = loss_fn(predictions, batch_y)
        return loss.item()