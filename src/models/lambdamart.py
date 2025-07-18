'''
Learn-to-rank model: LambdaMART
'''

import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRanker
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import time as time
import matplotlib.pyplot as plt
import seaborn as sns
from utils.util_metrics import topK_accuracy, calculate_MAP, calculate_MRR, helper_collections
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def lambdaMART(project_name, data_folder='data'):
    # Extracting the file path
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    file_name = f"{project_name}_features.csv"
    file_path = os.path.join(parent_dir, data_folder, file_name)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')

    # Load features.csv
    data = pd.read_csv(file_path)

    # Define feature columns and target
    feature_cols = ['rVSM_similarity', 'collab_filter', 'classname_similarity', 'bug_recency', 'bug_frequency']
    target_col = 'match'

    # kf = KFold(n_splits=10, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=10)
    acc_dicts, MAPs, MRRs = [], [], []

    bug_report_ids = data["report_id"].values  # Extract report_id for grouping

    for fold, (train_index, test_index) in enumerate(tqdm(gkf.split(data, groups=bug_report_ids), desc = 'Folds', unit=' fold')):
        start_time = time.time()  # Track time per fold
        print(f"Starting Fold {fold+1}/{gkf.get_n_splits()}")

        # Split the data into training and test sets
        X_train = data.iloc[train_index][feature_cols + ['report_id', 'file']]
        y_train = data.iloc[train_index][target_col]
        X_test = data.iloc[test_index][feature_cols + ['report_id', 'file']]
        y_test = data.iloc[test_index][target_col]
    
        # Preparing groups for ranking
        train_group = X_train['report_id'].value_counts().sort_index()

        # Drop 'report_id' and 'file' from features
        X_train_features = X_train.drop(columns=['report_id', 'file'])
        X_test_features = X_test.drop(columns=['report_id', 'file'])

        # print("\n X_train feature columns: ", X_train_features.columns)

        # Check if feature names match before prediction
        if not all(X_test_features.columns == X_train_features.columns):
            raise ValueError("Feature columns of test set do not match training set.")

        # Initialize LambdaMART model
        model = LGBMRanker(
            boosting_type = 'gbdt',
            objective = 'lambdarank',
            metric = 'mrr',
            n_estimators = 100, 
            learning_rate = 0.01,
            max_depth = 6,
            random_state = 42,
            verbose=-1  # Suppress LightGBM info messages
        )

        model.fit(X_train_features.values, y_train, group=train_group)

        # Prepare dictionaries for evaluation
        sample_dict, bug_reports, br2files_dict = helper_collections(X_test, project_name)

        # Calculate metrics
        acc_dict = topK_accuracy(bug_reports, sample_dict, br2files_dict, model)
        MAP = calculate_MAP(bug_reports, sample_dict, br2files_dict, model)
        MRR = calculate_MRR(bug_reports, sample_dict, br2files_dict, model)

        acc_dicts.append(acc_dict)
        MAPs.append(MAP)
        MRRs.append(MRR)

        fold_time = time.time() - start_time  # Time taken for this fold

        tqdm.write(f"Fold {fold+1}/{gkf.get_n_splits()} | X_train: {X_train.shape}, X_test: {X_test.shape} | Time: {fold_time:.2f}s")
    
    print("\n" + "="*40)
    print("Analyzing Feature Importance from Last Fold's Model")
    print("="*40)

    # The 'model' variable holds the trained model from the last fold
    # The 'feature_cols' variable holds the names of your features

    # 1. Extract feature importances and names
    feature_importance_values = model.feature_importances_

    # 2. Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importance_values
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importances:")
    print(importance_df)

    
    # Averaging metrics over all folds
    avg_acc_dict = {k: round(np.mean([d[k] for d in acc_dicts]).item(), 3) for k in acc_dicts[0].keys()}
    avg_MAP = round(np.mean(MAPs), 3)
    avg_MRR = round(np.mean(MRRs), 3)
    print('Average Top-K accuracy: ', avg_acc_dict)
    print('Average MAP: ', round(avg_MAP, 3))
    print('Average MRR: ', round(avg_MRR, 3))

    # # get feature importance
    # feature_importance = model.feature_importances_
    # feature_names = X_train.drop(columns=['report_id', 'file']).columns

    # # Sort and plot
    # sorted_idx = feature_importance.argsort()
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x=feature_importance[sorted_idx], y=feature_names[sorted_idx], palette='viridis')
    # plt.xlabel("Feature Importance Score")
    # plt.ylabel("Feature")
    # plt.title("Feature Importance in LightGBM")
    # plt.show()

    return "Done"
