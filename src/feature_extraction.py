'''
A Script for feature extraction

Notes: 
1. Eclipse_Platform_UI.txt it is taken from https://github.com/logpai/bughub/ -> https://github.com/logpai/bughub/tree/master/EclipsePlatform 
   where it is stored in eclipse_platform.zip.
2. Eclipse_Platform_UI.txt is a tsv file extracted by bughub owner from the xls file from the original dataset
3. Things to do: instead of having a .txt file directly gathered the data from .xls, so write a appropriate method in this after once the complete current setup is started running.

Update and enhancements: 7th Feb 2025
a) Buggy File Handling: Extracts before-fix versions from relevant commits.
b) Efficient Non-Buggy File Indexing: Extracts once at the latest before-fix commit.
c) Negative Sampling: Randomly selects 50 non-buggy files per bug report.
d) Commit Handling: Implements get_commit_before(timestamp) & checkout_code_at_timestamp(timestamp).
e) Logging & Debugging: Added output messages for tracking progress.

'''

from util import *;
from joblib import Parallel, delayed
import csv
import os
import time
from collections import defaultdict
from datetime import datetime
import subprocess

# Helper functions

class CodeTimer:
    def __init__(self, name="Code Timer"):
        self.name = name
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        print(f"{self.name} took {elapsed:.2f} seconds")

def get_commit_before(timestamp, repo_dir):
    '''
    Finds the latest commit (hash) in the repository that is before the given timestamp.

    Args:
        timestamp (int or float) : UNIX timestamp (e.g., from bug report commit_timestamp)
        repo_Dir (str): Path to the Git Repository

    Returns:
        str or None: Commit hash, or None if not found.
    '''
    # Convert UNIX timestamp to string "YYYY-MM-DD HH:MM:SS"
    date_str = datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
    cmd =["git", "rev-list", "-n", "1", "--before", date_str, "HEAD"]
    try:
        commit_hash = subprocess.check_output(cmd, shell=True, cwd=repo_dir).decode("utf-8").strip()
        return commit_hash
    except subprocess.CalledProcessError as e:
        print(f"Error finding commit for timestamp {date_str}: {e}")
        return None
    
def checkout_code_at_timestamp(timestamp, repo_dir):
    '''
    Checks out the repository at the commit corresponding to the given timestamp.

    Args:
        timestamp (int or float): UNIX timestamp.
        repo_dir (str) : Path to the Git repository

    Returns:
        str or None: The commit hash that was checked out or None if an error occured.
    
    '''
    commit_hash = get_commit_before(timestamp, repo_dir)
    if commit_hash:
        try:
            subprocess.run(["git", "checkout", commit_hash], check=True, cwd=repo_dir)
            return commit_hash
        except subprocess.CalledProcessError as e:
            return None
    else:
        return None


def extract(i, br, bug_reports, java_src_dict):
    '''
    Extracts features for 50 wrong (randomly chosen) files for each right (buggy) fie for the given bug report

    Arguments:
     i {integer} -- index for printing information
     br {dictionary} -- Given Bug Report
     bug_reports {list} -- List of all bug reports
     java_src_dict {dictionary} -- A dictionary of java source code (only the relevant files)

    Returns:
        list: List of featue rows
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
        
        except Exception as e:
            print(f"Error processing bug report {br_id} for file {java_file}: {e}")
            continue

    return features

def extract_features():
    '''
    Main pipeline for feature extraction. Clones the repository, reads bug reports,
    groups bug reports by commit, checks out the repository at the relevant commits,
    and extracts features only for the relevant Java files.
    '''

    # Keep time while extracting features
    with CodeTimer("Feature Extraction"):

        # Get the current directory (assuming the script is in the src folder)
        current_dir = os.path.dirname(__file__)

        # Navigate up one level from the src folder to reach the parent directory
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

        # Construct the path to the data folder
        data_folder_path = os.path.join(parent_dir, 'data')

         # Define the path to the repository's bundles folder (this can change when you checkout different commits)
        repo_dir = os.path.join(data_folder_path, "eclipse.platform.ui")
        repo_bundles_dir = os.path.join(repo_dir, 'bundles')


        # Clone git repo to a local folder
        git_clone(
            repo_url = "https://github.com/eclipse-platform/eclipse.platform.ui.git",
            clone_folder = data_folder_path,
        )

        # Read bug reports from tab separated file
        file_path = os.path.join(data_folder_path, 'Eclipse_Platform_UI.txt')
        bug_reports = tsv2dict(file_path)
        # print(bug_reports[:2]) # to veify the correctness


        # Sort bug reports by commit_timestamp (or report time) so that we can process in chronological order
        bug_reports.sort(key=lambda br: br["commit_timestamp"])

        # Group bug reports by the commit hash determined from each bug report timestamp.
        commit_groups = defaultdict(list)
        for br in bug_reports:
            commit_hash = get_commit_before(br["commit_timestamp"], repo_dir)
            if not commit_hash:
                print(f"No commit found for bug report ID {br['id']}. Skipping")
                continue
            commit_groups[commit_hash].append(br)
        
        print(f"Grouped bug reports into {len(commit_groups)} commit groups.")

        all_features = []
        indexed_non_buggy = {}

        # Process each group: checkout the commit once and then extract features for each bug report.
        for commit_hash, group in commit_groups.items():
            if not checkout_code_at_timestamp(group[0]["commit_timestamp"], repo_dir):
                continue

            buggy_files = set()
            for br in group:
                for file in br["files"]:
                    buggy_files.add(file)

            non_buggy_files = {
                f for f in get_all_java_file(repo_bundles_dir) if f not in buggy_files
            }
            indexed_non_buggy.update({f: read_file(f) for f in non_buggy_files})

            for br in group:
                features = extract_features_for_bug_report(br, buggy_files, indexed_non_buggy)
                all_features.extend(features)

            save_features_to_csv(all_features, os.path.join(data_folder_path, 'feature.csv'))

        # Save features to a csv file
        features_csv_path = os.path.join(data_folder_path, 'features.csv')
        features_csv_path = os.path.normpath(features_csv_path)
        with open(features_csv_path, "w", newline="") as f:
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

        print(f"Feature extraction complete. Features saved to {features_csv_path}")
    
        
