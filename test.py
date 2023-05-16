import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from model.q_model import QModel, QModel_Deep
from model.data_loader import GetDataLoaders
from dotenv import load_dotenv
import os
load_dotenv()

CMATRIX_DIR = os.getenv('CMATRIX_DIR')
MODEL_NAME = os.getenv('MODEL_NAME')
MODEL_WEIGHT_DIR = os.getenv('MODEL_WEIGHT_DIR')
CMATRIX_REPORT_DIR = os.getenv('CMATRIX_REPORT_DIR')
MODEL_TYPE = os.getenv('MODEL_TYPE')

def TestConfusionMatrix(model, test_dataloader, device):
    model.eval()
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            _, predicted_labels = torch.max(predictions, dim=1)
            all_predictions.extend(predicted_labels.cpu().numpy())
            all_true_labels.extend(batch_y.cpu().numpy())

    cm = confusion_matrix(all_true_labels, all_predictions)
    
    report = classification_report(all_true_labels, all_predictions, digits=4)
    # Write the report into a file
    with open(f'{CMATRIX_REPORT_DIR}/{MODEL_NAME}.txt', 'w') as f:
        f.write(report)
        
    return cm

def Test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = f'{MODEL_WEIGHT_DIR}/{MODEL_NAME}.pth'
    if MODEL_TYPE == 'deep':
        model = QModel_Deep().to(device)
    elif MODEL_TYPE == 'normal':
        model = QModel().to(device)
    model.load_state_dict(torch.load(model_path))
    _, test_dataloader = GetDataLoaders()
    cm = TestConfusionMatrix(model, test_dataloader, device)
    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'{CMATRIX_DIR}/{MODEL_NAME}.png', dpi=300)
    plt.clf()
