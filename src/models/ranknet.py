'''
bug localization model using RankNet
'''

import pandas as pd
import os
from itertools import combinations

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
        for buggy, non_buggy in zip(buggy_files.itertuples(index=False), non_buggy_files.itertuples(index=False)):
            pair = {
                'report_id' : report_id,
                'file1' : buggy.file,
                'file2' : non_buggy.file,
                'label' : 1 # Always 1 because file1 (buggy) should be ranked higher
            }

            for feature in feature_cols:
                pair[f"delta_{feature}"] = getattr(buggy, feature) - getattr(non_buggy, feature)

            pairwise_data.append(pair)

    return pd.DataFrame(pairwise_data)

def RankNet(data_folder='data', file_name='features.csv'):
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
    print(pairwise_data.head())
    print("pairwise data shape: ", pairwise_data.shape)


