from util import csv2dict, helper_collections, topK_accuracy, calculate_MAP, calculate_MRR
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

    # print(f"Sample Dict Keys: {list(sample_dict.keys())[:5]}")
    # print(f"Test Bug Reports IDs: {[br['id'] for br in bug_reports[-5:]]}")
    # print(f"BR2Files Dict: {dict(list(br2files_dict.items())[:5])}")

    # Calculating topk- 1, 5, 10 accuracy
    try:
        acc_dict = topK_accuracy(bug_reports, sample_dict, br2files_dict)
        MAP = calculate_MAP(bug_reports, sample_dict, br2files_dict)
        MRR = calculate_MRR(bug_reports, sample_dict, br2files_dict)
        print("Topk: ", acc_dict)
        print("MAP: ", MAP)
        print("MRR: ", MRR)
    except Exception as e:
        raise

    return "Done"