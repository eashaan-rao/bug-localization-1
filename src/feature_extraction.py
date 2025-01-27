'''
A Script for feature extraction

Notes: 
1. Eclipse_Platform_UI.txt it is taken from https://github.com/logpai/bughub/ -> https://github.com/logpai/bughub/tree/master/EclipsePlatform 
   where it is stored in eclipse_platform.zip.
2. Eclipse_Platform_UI.txt is a tsv file extracted by bughub owner from the xls file from the original dataset
3. Things to do: instead of having a .txt file directly gathered the data from .xls, so write a appropriate method in this after once the complete current setup is started running.
'''

from util import *;
from joblib import Parallel, delayed, cpu_count
import csv
import os

def extract(i, br, bug_reports, java_src_dict):
    '''
    Extracts features for 50 wrong (randomly chosen) files for each right (buggy) fie for the given bug report

    Arguments:
     i {integer} -- index for printing information
     br {dictionary} -- Given Bug Report
     bug_reports {list of dictionaries} -- All bug reports
     java_src_dict {dictionary} -- A dictionary of java source code
    '''

    print("Bug report: {}/{}".format(i + 1, len(bug_reports)), end="\r")

    br_id = br["id"]
    br_date = br["report_time"]
    br_files = br["files"]
    br_raw_text = br["raw_text"]

    features = []

    for java_file in br_files:
        java_file = os.path.normpath(java_file)

        try:
            # Source code of the JAVA file
            src = java_src_dict[java_file]

            # rVSM Text Similarity
            rvsm = cosine_sim(br_raw_text, src)

            # Class Name Similarity
            cns = class_name_similarity(br_raw_text, src)

            # Previous Reports
            prev_reports = previous_reports(java_file, br_date, bug_reports)

            # Collaborative Filter Score
            cfs = collaborative_filtering_score(br_raw_text, prev_reports)

            # Bug Fixing Recency
            bfr = bug_fixing_recency(br, prev_reports)

            # Bug Fixing Frequency
            bff = len(prev_reports)

            features.append([br_id, java_file, rvsm, cfs, cns, bfr, bff, 1])

            for java_file, rvsm, cns in top_k_wrong_files ( br_files, br_raw_text, java_src_dict):
                features.append([br_id, java_file, rvsm, cfs, cns, bfr, bff, 0])
        
        except:
            pass

    return features

def extract_features():
    '''
    Clones the git repository and parallelizes the feature extraction process
    '''

    # Keep time while extracting features
    with CodeTimer("Feature Extraction"):

        # Get the current directory (assuming the script is in the src folder)
        current_dir = os.path.dirname(__file__)

        # Navigate up one level from the src folder to reach the parent directory
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

        # Construct the path to the data folder
        data_folder_path = os.path.join(parent_dir, 'data')


        # Clone git repo to a local folder
        git_clone(
            repo_url = "https://github.com/eclipse-platform/eclipse.platform.ui.git",
            clone_folder = data_folder_path,
        )

        # Read bug reports from tab separated file
        file_path = os.path.join(data_folder_path, 'Eclipse_Platform_UI.txt')
        bug_reports = tsv2dict(file_path)

        # Read all java source files
        file_path = os.path.join(data_folder_path, 'eclipse.platform.ui/bundles/')
        java_src_dict = get_all_source_code(file_path)

        # Use all CPUs except one to speed up extraction and avoid computer lagging
        batches = Parallel(n_jobs=cpu_count() - 1) (
            delayed(extract)(i, br, bug_reports, java_src_dict)
            for i, br in enumerate(bug_reports)
        )

        # Flatten features
        features = [row for batch in batches for row in batch]

        # Save features to a csv file
        file_path = os.path.join(data_folder_path, 'features.csv')
        features_path = os.path.normpath(file_path)
        with open(features_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "report_id",
                    "file",
                    "rVSM_similarity",
                    "collab_filter",
                    "classname_similarity",
                    "bug_recency",
                    "bug_frequency",
                    "match",
                ]
            )
            for row in features:
                writer.writerow(row)

    
        
