'''
bug localization model using RankNet
'''

import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.util_metrics import topK_accuracy, calculate_MAP, calculate_MRR, helper_collections
from itertools import combinations

class RankNet(nn.Module):
    def __init__(self, input_dim):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Output probability P(file1 > file2)
        )
    
    def forward(self, x):
        return self.model(x)
    
class EarlyStopping:
    '''
    Early stops training if validation loss doesn't improve after a given patience.
    '''
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()  # save best model state
        else:
            self.counter += 1
            if self.counter >- self.patience:
                self.early_stop = True

    
def ranknet_loss(y_pred, y_true):
    '''
    Implements the RankNet pairwise loss.
    '''
    # P(file1 > file2) = sigmoid(s1 - s2)
    diff = y_pred[:, 0] - y_pred[:, 1] # s1 - s2
    prob = torch.sigmoid(diff)
    loss = -torch.mean(y_true * torch.log(prob + 1e-10) + (1 - y_true) * torch.log(1- prob + 1e-10))
    return loss

def transform_to_pairwise(data):
    '''
    Transforms pointwise ranking data into pairwise formate for RankNet.

    Arguments:
        data (pd.DataFrame): DataFrame with 'report_id', 'file', feature columns and 'label'

    Returns:
        pd.DataFrame: Pairwise transformed dataset with feature differences.
    '''

    pairwise_data = []
    feature_cols = [col for col in data.columns if col not in ['report_id', 'file', 'match']]

    for report_id, group in data.groupby("report_id"):
        buggy_files = group[group['match'] == 1]
        non_buggy_files = group[group['match'] == 0]

        # Generate pairs (buggy, non-buggy) and compute feature differences
        for buggy in buggy_files.itertuples(index=False):
            for non_buggy in non_buggy_files.itertuples(index=False):
                pair1 = {
                    'report_id' : report_id,
                    'file1' : buggy.file,
                    'file2' : non_buggy.file,
                    'label' : 1 # Always 1 because file1 (buggy) should be ranked higher
                }

                for feature in feature_cols:
                    pair1[f"delta_{feature}"] = getattr(buggy, feature) - getattr(non_buggy, feature)
                pairwise_data.append(pair1)

                # Reverse pair (non-buggy ranked higher) -> label 0
                pair2 = {
                    'report_id' : report_id,
                    'file1' : non_buggy.file,
                    'file2' : buggy.file,
                    'label' : 0 # Always 0 because file1 (buggy) is ranked lower
                }

                for feature in feature_cols:
                    pair2[f"delta_{feature}"] = getattr(buggy, feature) - getattr(non_buggy, feature)
                pairwise_data.append(pair2)

    return pd.DataFrame(pairwise_data)

def rankNet(data_folder='data', file_name='features.csv'):
    '''
    Trains and evaluates RankNet for bug localization using PyTorch.
    '''

    # Extracting the file path
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    file_path = os.path.join(parent_dir, data_folder, file_name)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')

    # Load features.csv
    data = pd.read_csv(file_path)

    # Dataset transformation
    pairwise_data = transform_to_pairwise(data)
    print("pairwise data shape: ", pairwise_data.shape)

    return pairwise_data

    # # Defime feature columns and target
    # feature_cols = [
    #     'delta_rVSM_similarity', 'delta_collab_filter',
    #     'delta_classname_similarity', 'delta_bug_recency', 'delta_bug_frequency'
    # ]
    # target_col = 'label'

    # # GroupKFold for 10-fold cross-validation
    # gkf = GroupKFold(n_splits=10)
    # acc_dicts, MAPs, MRRs = [], [], []

    # bug_report_ids = pairwise_data["report_id"].values # Extract report_id for grouping

    # for fold, (train_index, test_index) in enumerate(tqdm(gkf.split(pairwise_data, groups=bug_report_ids), desc='Folds', unit=' fold')):
    #     start_time = time.time()
    #     print(f"Starting Fold {fold+1}/{gkf.get_n_splits()}")

    #     # Split the data intro train and test sets
    #     X_train = pairwise_data.iloc[train_index][feature_cols].values.astype(np.float32)
    #     y_train = pairwise_data.iloc[train_index][target_col].values.astype(np.float32)
    #     X_test = pairwise_data.iloc[test_index][feature_cols].values.astype(np.float32)
    #     y_test = pairwise_data.iloc[test_index][target_col].values.astype(np.float32)
    #     X_test_df = pairwise_data.iloc[test_index][['report_id', 'file1', 'file2'] + feature_cols]

    #     # Convert to PyTorch tensors
    #     X_train_tensor = torch.tensor(X_train)
    #     y_train_tensor = torch.tensor(y_train).view(-1, 1)
    #     X_test_tensor = torch.tensor(X_test)

    #     epochs = 10
    #     batch_size = 32
    #     learning_rate = 0.001

    #     # Create DataLoader for training
    #     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #     # Define model, optimizer, and loss
    #     ranknet_model = RankNet(input_dim=len(feature_cols))
    #     optimizer = optim.Adam(ranknet_model.parameters(), lr = learning_rate)
    #     early_stopping = EarlyStopping(patience=3, delta=0.001)

    #     # Training loop
    #     ranknet_model.train()
    #     for epoch in range(epochs):
    #         total_loss = 0.0
    #         for X_batch, y_batch in train_loader:
    #             optimizer.zero_grad()
    #             preds = ranknet_model(X_batch)

    #             # Create pairwise combinations
    #             s1 = preds[::2]  # file1 scores
    #             s2 = preds[1::2]  # file2 scores
    #             y_true = y_batch[::2]   # Ground truth

    #             y_pred = torch.cat([s1, s2], dim=1)
    #             loss = ranknet_loss(y_pred, y_true)

    #             loss.backward()
    #             optimizer.step()
    #             total_loss += loss.item()

    #         avg_loss = total_loss / len(train_loader)
    #         print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {avg_loss:.4f}")

    #         # Validation phase - compute validation loss for early stopping
    #         ranknet_model.eval()
    #         val_loss = 0.0
    #         with torch.no_grad():
    #             for X_val, y_val in train_loader: # Using training data for validation as  pairwise data
    #                 val_preds = ranknet_model(X_val)
    #                 s1_val = val_preds[::2]
    #                 s2_val = val_preds[1::2]
    #                 y_val_true = y_val[::2]
    #                 y_val_pred = torch.cat([s1_val, s2_val], dim=1)
    #                 val_loss += ranknet_loss(y_val_pred, y_val_true).item()
            
    #         avg_val_loss = val_loss / len(train_loader)
    #         print(f"Validation Loss: {avg_val_loss:.4f}")

    #         # Call early stopping
    #         early_stopping(avg_val_loss, ranknet_model)

    #         if early_stopping.early_stop:
    #             print(f"Early stopping triggered at epoch {epoch+1} for fold {fold+1}")
    #             ranknet_model.load_state_dict(early_stopping.best_model_state) # load best model
    #             break

    #     # # Prepare sample_dict and br2files_dict for evaluation
    #     # sample_dict, bug_reports, br2files_dict = helper_collections(X_test_df)

    #     # # Generate predictions for all pairs in the test set
    #     # ranknet_model.eval()
    #     # with torch.no_grad():
    #     #     predictions = ranknet_model(X_test_tensor).flatten().numpy()

    #     # # create a dictionary for relevancy scores to use in evaluation
    #     # relevancy_dict = {}
    #     # for i, row in X_test_df.iterrows():
    #     #     bug_id = row['report_id']
    #     #     file1, file2 = row['file1'], row['file2']

    #     #     # if file1 has a higher probability, assign higher relevancy score
    #     #     if bug_id not in relevancy_dict:
    #     #         relevancy_dict[bug_id] = {}

    #     #     relevancy_dict[bug_id][file1] = relevancy_dict[bug_id].get(file1, 0) +  predictions[i]
    #     #     relevancy_dict[bug_id][file2] = relevancy_dict[bug_id].get(file2, 0) +  (1- predictions[i])

    #     # # Calculate metrics
    #     # acc_dict = topK_accuracy(bug_reports, sample_dict, br2files_dict, relevancy_dict)
    #     # MAP = calculate_MAP(bug_reports, sample_dict, br2files_dict, relevancy_dict)
    #     # MRR = calculate_MRR(bug_reports, sample_dict, br2files_dict, relevancy_dict)

    #     # acc_dicts.append(acc_dict)
    #     # MAPs.append(MAP)
    #     # MRRs.append(MRR)

    #     # fold_time = time.time() - start_time
    #     # tqdm.write(f"Fold {fold+1}/{gkf.get_n_splits()} | X_train: {X_train.shape}, X_test: {X_test.shape} | Time: {fold_time:.2f}s")

    # # # Averaging metrics over all folds
    # # avg_acc_dict = {k: round(np.mean([d[k] for d in acc_dicts]).item(), 3) for k in acc_dicts[0].keys()}
    # # avg_MAP = round(np.mean(MAPs), 3)
    # # avg_MRR = round(np.mean(MRRs), 3)
    # # print('Average top-k Accuracy: ', avg_acc_dict)
    # # print('Average MAP: ', avg_MAP)
    # # print('Average MRR: ', avg_MRR)

    # return "Done"




