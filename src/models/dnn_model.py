from utils.util_metrics import helper_collections, topK_accuracy, calculate_MAP, calculate_MRR
from sklearn.neural_network import MLPRegressor
from imblearn.over_sampling import RandomOverSampler
from joblib import Parallel, delayed
from math import ceil
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def oversample(train_samples):
    '''
    Oversamples the minority class using RandomOverSampler

    Arguments:
        train_samples {pd.DataFrame} -- DataFrame containing training samples with feature columns
                                        and a 'match' column as  the label

    Returns:
        pd.DataFrame -- Oversampled DataFrame with balances 'match' labels
    '''
    
    
    X = train_samples.drop(columns=['match']) # Features excluding the label column
    y = train_samples['match'] # label column

    # print("X.shape: ", X.shape)
    # print("y.shape: ", y.shape)

    # Applu random oversampling to balance the 'match' column
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # # oversample features of buggy files (note: didn't understand the logic)
    # for i, sample in enumerate(samples):
    #     samples_.append(sample)
    #     if i % 51 == 0:
    #         for _ in range(9):
    #             samples_.append(sample)

    # Reconstruct the DataaFrame with the 'match' column added back
    oversampled_train_samples = X_resampled.copy()
    oversampled_train_samples['match'] = y_resampled
    
    # print("Oversampled train samples shape: ", oversampled_train_samples.shape)

    return oversampled_train_samples

def features_and_labels(samples):
    '''
    Returns features and labels for the given dataframe of samples

    Arguments:
        samples {pd.Dataframe} -- samples from features.csv
    
        Returns:
            X {np.ndarray} -- Feature array of shape(n_samples, n_features).
            y {np.ndarray} -- Label array of shape (n_samples,).
    '''

    # Extract features and labels directly using pandas
    X = samples[['rVSM_similarity', 'collab_filter', 'classname_similarity', 'bug_recency', 'bug_frequency']].astype(float).values
    y = samples["match"].astype(float).values

    # num_samples = len(samples)
    # X = np.zeros((num_samples, 5))
    # y = np.zeros((num_samples, 1))

    # for i, sample in enumerate(samples):
    #     X[i][0] = float(sample['rVSM_similarity'])
    #     X[i][1] = float(sample["collab_filter"])
    #     X[i][2] = float(sample["classname_similarity"])
    #     X[i][3] = float(sample["bug_recency"])
    #     X[i][4] = float(sample["bug_frequency"])
    #     y[i] = float(sample["match"])

    return X, y

def kfold_split_indexes(k, len_samples):
    '''
    Returns list of tuples for split start(inclusive) and finish(exclusive) indexes.

    Arguments:
        k {integer} -- the number of folds
        len_samples {integer} -- the length of the sample list
    '''

    step = ceil(len_samples / k)
    ret_list = [(start, start + step) for start in range(0, len_samples, step)]

    return ret_list

def kfold_split(bug_reports, samples, start, finish):
    '''
    Returns train samples and bug reports for test

    Arguments:
        bug_reports {list of dictionaries} -- list of all bug reports
        samples {list} -- samples from features.csv
        start {integer} -- start index from test fold
        finish {integer} -- start index for test fold
    '''

    # Splitting train and test samples
    train_samples = pd.concat([samples.iloc[:start], samples.iloc[finish:]], ignore_index=True)
    test_samples = samples.iloc[start:finish]

    # Extracting report IDs and matching with bug reports
    test_br_ids = set(test_samples['report_id'].astype(int))
    test_bug_reports = [br for br in bug_reports if int(br["id"]) in test_br_ids]

    return train_samples, test_bug_reports

def train_dnn(i, num_folds, df, start, end, sample_dict, bug_reports, br2files_dict):
    '''
    Trains the DNN model and calculates top-k accuracies

    Arguments:
    i {integer} -- current fold number for printing information
    num_folds {integer} -- Total number of folds
    df {Dataframe} -- Dataframe of samples from features.csv
    start {integer} -- start index for test fold
    end {integer} -- start/last index for test fold
    sample_dict {dict} -- Dictionary of all bug reports
    br2files_dict {dict} -- Bug report ID related files mapping
    '''

    print(f"Fold: {i + 1} / {num_folds}", end="\r")

    train_samples, test_bug_reports = kfold_split(bug_reports, df, start, end)
    train_samples = oversample(train_samples)
    train_samples = train_samples.sample(frac=1, random_state=42).reset_index(drop=True)
    X_train, y_train = features_and_labels(train_samples)
    # print(len(X_train))
    # print(len(y_train))

    clf = MLPRegressor(
        solver= "sgd",
        alpha=1e-5,
        hidden_layer_sizes=(300,),
        random_state=1,
        max_iter=200,
        n_iter_no_change=10,
        tol=1e-3,
        verbose=False
    )

    # Show a progress bar for the folds
    for i in tqdm(range(num_folds), desc="Folds", unit="fold"):
        print(f"Training Fold {i + 1} / {num_folds}", flush=True)
        clf.fit(X_train, y_train.ravel())

    acc_dict = topK_accuracy(test_bug_reports, sample_dict, br2files_dict, clf=clf)
    MAP = calculate_MAP(test_bug_reports, sample_dict, br2files_dict, clf=clf)
    MRR = calculate_MRR(test_bug_reports, sample_dict, br2files_dict, clf=clf)
    return acc_dict, MAP, MRR

def dnn_model_kfold(project_name, k=10, data_folder='data', n_jobs=-2, random_state=42):
    '''
    Run k-fold cross validation in parallel

    Arguments:
    project_name {str} -- the name of project under consideration.
    k {integer} -- the number of folds (default: {10})
    data_folder {str} -- folder containing the features file
    n_jobs {integer} -- number of CPU cores to use (default: all but one)
    random_state {integer} -- random state for reproducibility (default: 42)
    '''
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    file_name = f"{project_name}_features.csv"
    file_path = os.path.join(parent_dir, data_folder, file_name)
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')
    
    df = pd.read_csv(file_path)

    # These collections are speed up the process while calculating top-k accuracy
    sample_dict, bug_reports, br2files_dict = helper_collections(df)

    # Ensure reproducible shuffling
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Calculate fold sizes
    step = int(np.ceil(len(df) / k))
    fold_indices = [(i, i + step) for i in range(0, len(df), step)]

    # print("step: ", step)
    # print("fold_indices: ", fold_indices)

    # K-fold cross validation in parallel
    results = Parallel(n_jobs=n_jobs) (
        # Uses all cores but one
        delayed(train_dnn) (i, k, df, start, end, sample_dict, bug_reports, br2files_dict)
        for i, (start, end) in enumerate(fold_indices) 
    )

    # Separate results into accuracy dictionaries, MAPs and MRRs
    acc_dicts, MAPs, MRRs = zip(*results)

    # Calculating the average accuracy from all folds
    avg_acc_dict = {
        key: round(np.mean([d[key] for d in acc_dicts]), 3) for key in acc_dicts[0].keys()
    }

    # Calculate the average MAP and MRR across all folds
    avg_MAP = round(np.mean(MAPs), 3)
    avg_MRR = round(np.mean(MRRs), 3)

    print("Average Top-K accuracy: ", avg_acc_dict)
    print("Average MAP: ", avg_MAP)
    print("Average MRR: ", avg_MRR)
    # for key in acc_dicts[0].keys():
    #     avg_acc_dict[key] = round(sum([d[key] for d in acc_dicts]) / len(acc_dicts), 3)
    
    return avg_acc_dict, avg_MAP, avg_MRR