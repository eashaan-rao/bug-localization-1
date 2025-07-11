'''
Training a bug localization model made with DNN with custom loss
'''

# Import Libraries
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.util_metrics import helper_collections, unified_topK_MAP_MRR

# Device Configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DNN model for bug localization
class BugLocalizationDNN(nn.Module):
    def __init__(self, input_dim):
        super(BugLocalizationDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)  # BatchNorm Layer 1
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)   # BatchNorm Layer 2
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)   # BatchNorm Layer 3
        self.output = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.output(x))
        return x
    
# Focal loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, rank_weighting = True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.rank_weighting = rank_weighting

    def forward(self, inputs, targets, ranks=None):
        '''
        inputs: predicted probabilities (after sigmoid) - shape (batch_size,)
        targets: ground truth labels - shape (batch_size,)
        ranks: rank positins of predictions - shape (batch_size,)
        '''
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        # Apply rank-based penalty if enabled
        if self.rank_weighting and ranks is not None:
            rank_penalty =  1 / torch.log2(2 + ranks.float())  # Penalized lower-ranked items
            focal_loss *= rank_penalty

        return focal_loss.mean()
    
# Main DNN Training Function
def train_dnn_with_custom_loss(project_name, data_folder = 'data'):
    # Load Data
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    file_name = f"{project_name}_features.csv"
    file_path = os.path.join(parent_dir, data_folder, file_name)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'file note found: {file_path}')
    
    # Load and preprocess data
    data = pd.read_csv(file_path)
    feature_cols = ['rVSM_similarity', 'collab_filter', 'classname_similarity', 'bug_recency', 'bug_frequency']
    target_col = 'match'

    # Define GroupKFold for splitting
    gkf = GroupKFold(n_splits=10)
    bug_report_ids = data['report_id'].values

    acc_dicts, MAPs, MRRs = [], [], []

    #Initialize to store training and test losses
    train_losses_per_fold = []
    test_losses_per_fold = []

    for fold, (train_index, test_index) in enumerate(tqdm(gkf.split(data, groups= bug_report_ids), desc='Folds', unit=' fold')):
        start_time = time.time()
        print(f"Starting Fold {fold+1}/{gkf.get_n_splits()}")

        # Reset losses for this fold
        train_losses = []
        test_losses = []

        # Split the data into train and test
        X_train = data.iloc[train_index][feature_cols].values.astype(np.float32)
        y_train = data.iloc[train_index][target_col].values.astype(np.float32)
        X_test = data.iloc[test_index][feature_cols].values.astype(np.float32)
        y_test = data.iloc[test_index][target_col].values.astype(np.float32)

        X_test_df = data.iloc[test_index][['report_id', 'file'] + feature_cols]

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train).to(device)
        y_train_tensor = torch.tensor(y_train).view(-1, 1).to(device)
        X_test_tensor = torch.tensor(X_test).to(device)
        y_test_tensor = torch.tensor(y_test).to(device)

        # print(f"X_test_tensor shape: {X_test_tensor.shape}")
        # print(f"y_test_tensor shape: {y_test_tensor.shape}")

        # Prepare DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Define Model, loss and optimizer
        model = BugLocalizationDNN(input_dim=len(feature_cols)).to(device)
        criterion = FocalLoss(alpha=0.25, gamma=2.0, rank_weighting=True) # Focal Loss as custom loss
        # adding L2 regularization (Weight decay)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # Training Loop
        epochs = 10
        patience = 3
        best_loss = np.inf
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            for batch_idx,(X_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                y_batch = y_batch.squeeze()
                # forward pass
                preds = model(X_batch).squeeze() # shape: (batch_size,)
                if preds.dim() > 1:
                    preds = preds.squeeze()
                if preds.dim() == 0:
                    preds = preds.unsqueeze(0)  # make it (1,)
                if y_batch.dim()==0:
                    y_batch = y_batch.unsqueeze(0) # make it (1,)
                # Generate rank positions for current batch
                if preds.dim() > 0 and len(preds) > 0:
                    ranks = torch.arange(1, len(preds) + 1).to(preds.device)
                else:
                    ranks = torch.tensor([1.0], device=preds.device)  # Handle scalar or 0-D tensor
                # calculate loss with rank-aware penalty
                loss = criterion(preds, y_batch, ranks)
                # backpropagation and optimization
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                

            # Store training loss for each epoch
            train_losses.append(total_loss)
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {avg_loss:.4f}")

            # Evaluation on the test set
            model.eval()
            with torch.no_grad():
                y_test_pred = model(X_test_tensor).squeeze() # Shape: (num_test_samples,)

                # generate ranks for test predictions
                ranks_test = torch.arange(1, len(y_test_pred) + 1).to(y_test_pred.device)

                # Reshape target tensor to match the prediction shape
                y_test_tensor = y_test_tensor.view(-1, 1)
                y_test_tensor = y_test_tensor.squeeze()

                #calculate test loss with rank penalty
                test_loss = criterion(y_test_pred, y_test_tensor, ranks_test)
                test_losses.append(test_loss.item())
            
            # Early stopping logic
            if avg_loss < best_loss - 0.001:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break
        
        # Evaluation Phase
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor).flatten().cpu().numpy()

        # print(f"Shape of X_test_df: {X_test_df.shape}")
        # print(f"Length of predictions: {len(predictions)}")

        # exit(0)

        # Prepare relevancy dictionary for evaluation
        relevancy_dict = {}
        for i, row in enumerate(X_test_df.itertuples(index=False)):
            bug_id, file = row.report_id, row.file
            if bug_id not in relevancy_dict:
                relevancy_dict[bug_id] = {}
            relevancy_dict[bug_id][file] = predictions[i]

        # Calculate metrics
        sample_dict, bug_reports, br2files_dict = helper_collections(X_test_df, project_name)
        acc_dict, MAP, MRR = unified_topK_MAP_MRR(
            bug_reports,
            sample_dict,
            br2files_dict,
            relevancy_dict = relevancy_dict,  # pass the precomputed scores here
            clf = None,
            k_values = [1, 5, 10]
        )

        acc_dicts.append(acc_dict)
        MAPs.append(MAP)
        MRRs.append(MRR)

        fold_time = time.time() - start_time
        tqdm.write(f"Fold {fold+1}/{gkf.get_n_splits()} | X_train: {X_train.shape}, X_test: {X_test.shape} | Time: {fold_time:.2f}s")

        # Store losses for each fold
        train_losses_per_fold.append(train_losses)
        test_losses_per_fold.append(test_losses)

    
    # Plot training and test loss curves for each fold
    plt.figure(figsize=(12, 8))
    for fold in range(10):
        actual_epochs = len(train_losses_per_fold[fold])  # Get the actual epochs
        plt.plot(range(1, actual_epochs + 1), train_losses_per_fold[fold], label=f"Train Loss (Fold {fold + 1})", linestyle='--')
        plt.plot(range(1, actual_epochs + 1), test_losses_per_fold[fold], label=f"Test Loss (Fold {fold + 1})", linestyle='-')
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs. Test Loss Across Folds")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate Final Metrics
    avg_acc_dict = {k: round(np.mean([d[k] for d in acc_dicts]), 3) for k in acc_dicts[0].keys()}
    avg_MAP = round(np.mean(MAPs), 3)
    avg_MRR = round(np.mean(MRRs), 3)

    print('Average Top-K accuracy: ', avg_acc_dict)
    print('Average MAP: ', avg_MAP)
    print('Average MRR: ', avg_MRR)

    return "Done"

