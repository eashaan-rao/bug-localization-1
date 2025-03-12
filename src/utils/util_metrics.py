'''
A helper file contanining methods to calculate all the metrics for a bug localization model.
'''

import os
import numpy as np
from utils.util import tsv2dict
from lightgbm import LGBMRanker

def helper_collections(samples_df, only_rvsm=False):
    '''
    Generates helper function for calculations

    Arguments:
        samples_df {pd.DataFrame} -- Dataframe containing samples from features.csv

    keyword Arguments:
        only_rvsm {bool} -- If True only 'rvsm' features are added to 'sample_dict'. (default: {False}) 
    '''

    sample_dict = {}
    # Initialize sample_dict with empty lists for each unique report_id
    for report_id in samples_df["report_id"].unique():
        sample_dict[report_id] = []

    # Iterate through the DataFrame rows
    for _, row in samples_df.iterrows():
        temp_dict = {}
        values = [float(row["rVSM_similarity"])]
        if not only_rvsm:
            values += [
                float(row["collab_filter"]),
                float(row["classname_similarity"]),
                float(row["bug_recency"]),
                float(row["bug_frequency"]),
            ]
        temp_dict[os.path.normpath(row["file"])] = values

        sample_dict[row["report_id"]].append(temp_dict)

    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    data_folder_path = os.path.join(parent_dir, 'data')
    file_path = os.path.join(data_folder_path, 'Eclipse_Platform_UI.txt')
    
    bug_reports = tsv2dict(file_path)
    br2files_dict = {}

    for bug_report in bug_reports:
        br2files_dict[int(bug_report["id"])] = bug_report["files"]
    
    return sample_dict, bug_reports, br2files_dict

def topK_accuracy(test_bug_reports, sample_dict, br2files_dict, clf=None):
    '''
    Calculates top-k accuracies for specified k valies (e.g. top-1, top-5, top-10)

    Arguments:
        test_bug_reports {list of dictionaries} -- list of all bug reports
        sample_dict {dictionary of dictionaries} -- a helper collection for fast accuracy calculation
        br2files_dict {dictionary} -- dictionary for "bug report id - list of all related files in features.csv pairs

    keyword arguments:
        clf {object} -- A classifier with 'predict()' function. If none, rvsm relevancy is used (default: None)
    
    Returns:
        dict -- Dictionary with top-k values as keys and accuracy as values
    '''

    top_k = [1, 5, 10]
    # topk_counters = [0] * 20
    topk_counters = {k: 0 for k in top_k}
    negative_total = 0
    for bug_report in test_bug_reports:
        dnn_input = []
        corresponding_files = []
        bug_id = int(bug_report["id"])

        try:
            for temp_dict in sample_dict[bug_id]:
                java_file = list(temp_dict.keys())[0]
                features_for_java_file = list(temp_dict.values())[0]
                dnn_input.append(features_for_java_file)
                corresponding_files.append(java_file)
        except KeyError:
            negative_total += 1
            continue
        
        # Calculate relevancy for all files related to the bug report in features.csv
        # In features.csv, there are 50 wrong(randomly choosen) files for each right (buggy) file.
        relevancy_list = []

        # Define feature columns
        feature_cols = ['rVSM_similarity', 'collab_filter', 'classname_similarity', 'bug_recency', 'bug_frequency']

        if isinstance(clf, LGBMRanker):
            dnn_input_np = np.array(dnn_input)
            relevancy_list = clf.predict(dnn_input_np)
        elif clf: # dnn classifier
            relevancy_list = clf.predict(dnn_input)
        else: # rVSM
            relevancy_list = np.array(dnn_input).ravel()

        # Top-1, top-2 .. top-20 accuracy
        for k in top_k:
            # max_indices = np.argpartition(relevancy_list, -i)[-i:]
            max_indices = np.argsort(relevancy_list)[-k:]
            for corresponding_file in np.array(corresponding_files)[max_indices]:
                if str(corresponding_file) in br2files_dict[bug_id]:
                    topk_counters[k] += 1
                    break
    
    acc_dict = {}
    denominator = len(test_bug_reports) - negative_total
    if denominator == 0:
        print("Warning: No valid bug reports found in sample_dict. Returning empty accuracy dictionary.")
        return acc_dict # Return an empty dictionary to prevent the error
    
    for k, counter in topk_counters.items():
        acc_dict[k] = round(counter / denominator, 3)

    return acc_dict

def calculate_MAP(test_bug_reports, sample_dict, br2files_dict, clf=None):
    '''
    Calculaates Mean Average Precision (MAP)
    '''
    average_precisions = []
    for bug_report in test_bug_reports:
        dnn_input = []
        corresponding_files = []
        bug_id = int(bug_report["id"])

        try:
            for temp_dict in sample_dict[bug_id]:
                java_file = list(temp_dict.keys())[0]
                features_for_java_file = list(temp_dict.values())[0]
                dnn_input.append(features_for_java_file)
                corresponding_files.append(java_file)
        except KeyError:
            continue

        if isinstance(clf, LGBMRanker):
            dnn_input_np = np.array(dnn_input)
            relevancy_list = clf.predict(dnn_input_np)
        elif clf: # dnn classifier
            relevancy_list = clf.predict(dnn_input)
        else: # rVSM
            relevancy_list = np.array(dnn_input).ravel()

        ranked_indices = np.argsort(relevancy_list)[::-1]

        relevant_files = br2files_dict.get(bug_id, [])
        hits, precision_at_i, rank = 0, 0.0, 0

        for idx in ranked_indices:
            rank += 1
            if str(corresponding_files[idx]) in relevant_files:
                hits += 1
                precision_at_i += hits/rank
            if hits > 0:
                average_precisions.append(precision_at_i / hits)
    return round(np.mean(average_precisions), 3) if average_precisions else 0.0 

def calculate_MRR(test_bug_reports, sample_dict, br2files_dict, clf=None):
    '''
    Calculates Mean Reciprocal Rank (MRR)
    '''
    reciprocal_ranks = []
    for bug_report in test_bug_reports:
        dnn_input = []
        corresponding_files = []
        bug_id = int(bug_report["id"])

        try:
            for temp_dict in sample_dict[bug_id]:
                java_file = list(temp_dict.keys())[0]
                features_for_java_file = list(temp_dict.values())[0]
                dnn_input.append(features_for_java_file)
                corresponding_files.append(java_file)
        except KeyError:
            continue
        
        if isinstance(clf, LGBMRanker):
            dnn_input_np = np.array(dnn_input)
            relevancy_list = clf.predict(dnn_input_np)
        elif clf: # dnn classifier
            relevancy_list = clf.predict(dnn_input)
        else: # rVSM
            relevancy_list = np.array(dnn_input).ravel()

        ranked_indices = np.argsort(relevancy_list)[::-1]

        relevant_files = br2files_dict.get(bug_id, [])
        
        for rank, idx in enumerate(ranked_indices, start=1):
            if str(corresponding_files[idx]) in relevant_files:
                reciprocal_ranks.append(1.0 / rank)
                break
    return round(np.mean(reciprocal_ranks), 3) if reciprocal_ranks else 0.0
