from util import csv2dict, helper_collections, topK_accuracy
import pandas as pd
import os



def rvsm_model(data_folder_path=None):
    if data_folder_path is None:
        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        data_folder_path = os.path.join(parent_dir, 'data')
    
    file_path = os.path.join(data_folder_path, 'features.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    if "rVSM_similarity" not in df.columns:
        raise ValueError(f"Missing 'rVSM_similarity' column in {file_path}")
    
    # samples = csv2dict(file_path)
    rvsm_list = df["rVSM_similarity"].astype(float).tolist()

    # These collections are speed up the process while calculating top-k accuracy
    try:
        sample_dict, bug_reports, br2files_dict = helper_collections(df, True)
    except Exception as e:
        raise

    # Calculating topk accuracy
    try:
        acc_dict = topK_accuracy(bug_reports, sample_dict, br2files_dict)
    except Exception as e:
        raise

    return acc_dict