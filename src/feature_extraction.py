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
import time
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
    cmd = f'git log --before="{date_str}" --pretty=format:"%H" -n 1'
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


        # Clone git repo to a local folder
        git_clone(
            repo_url = "https://github.com/eclipse-platform/eclipse.platform.ui.git",
            clone_folder = data_folder_path,
        )

        # Read bug reports from tab separated file
        file_path = os.path.join(data_folder_path, 'Eclipse_Platform_UI.txt')
        bug_reports = tsv2dict(file_path)
        print(bug_reports[:2]) # to veify the correctness

        # Sort bug reports by commit_timestamp (or report time) so that we can process in chronological order
        bug_reports.sort(key=lambda br: br["commit_timestamp"])

        # Define the path to the repository's bundles folder (this can change when you checkout different commits)
        repo_dir = os.path.join(data_folder_path, "eclipse.platform.ui")
        repo_bundles_dir = os.path.join(repo_dir, 'bundles')

        features = []

        # Group bug reports by the commit hash determined from each bug report timestamp.
        commit_groups = {}
        for br in bug_reports:
            ts = br["commit_timestamp"]
            commit_hash = get_commit_before(ts, repo_dir)
            if not commit_hash:
                print(f"No commit found for bug report ID {br['id']}. Skipping")
                continue
            commit_groups.setdefault(commit_hash, []).append(br)
        
        print(f"Grouped bug reports into {len(commit_groups)} commit groups.")

        # Process each group: checkout the commit once and then extract features for each bug report.
        for commit_hash, group in commit_groups.items():
            print(f"Processing group for commit {commit_hash} with {len(group)} bug reports.")
            try:
                subprocess.run(["git", "checkout", commit_hash], check=True, cwd=repo_dir)
            except subprocess.CalledProcessError as e:
                print(f"Error checking out commit {commit_hash}: {e}")
                continue

            # For each bug report in this commit group:
            for i, br in enumerate(group):
                # Extract only the relevant Java files for this bug report.
                java_src_dict = get_relevant_source_code(br["files"], repo_bundles_dir)
                features.extend(extract(i, br, bug_reports, java_src_dict))


        '''
        Commenting this part of the code till I make further changes...
        '''
        # current_commit = None # To track which commit is currently checked out
        # current_java_src_dict = None # Source code for the currently checked out commit

        # for i, br in enumerate(bug_reports):
        #     # Get the commit timestamp for the bug report (You can also use br['report_time'])
        #     bug_ts = br["commit_timestamp"]
        #     # Determine the desired commit hash for this bug report's timestamp
        #     desired_commit = get_commit_before(bug_ts)

        #     # If the desired commit differs from the current one, cheout the new commit.
        #     if desired_commit != current_commit:
        #         checkout_code_at_timestamp(bug_ts)
        #         cuurent_commit = desired_commit
        #         # Reset the current source dictionary; we will re-read the files for this bug report
        #         current_java_src_dict = get_relevant_source_code(br["files"], repo_bundles_dir)

        #     # For this bug report, extract only the relevant Java files using the current repo snapshot.
        #     # br["files"] should be alist of relative file paths that match the keys in the repository.
        #     # Now extract features for this bug report using only the relevant files
        #     feat = extract(i, br, bug_reports, current_java_src_dict)
        #     features.extend(feat)


        # # Read all java source files
        # file_path = os.path.join(data_folder_path, 'eclipse.platform.ui/bundles/')
        # java_src_dict = get_all_source_code(file_path)

        # print(list(java_src_dict.keys())[:5]) # print first 5 keys to verify paths

        # # Print the first 5 keys and a short snippet of their content
        # for key in list(java_src_dict.keys())[:5]:
        #     print(f"File: {key}\nContent: {java_src_dict[key][:200]}...\n")

        # # Check total files count
        # print(f"Total Java files found: {len(java_src_dict)}")

        # # Pick a file from bug_reports and check if it exists in java_src_dict
        # sample_bug_report_file = bug_reports[0]["files"][0] # First file in first bug report

        # if sample_bug_report_file in java_src_dict:
        #     print(f"Found! {sample_bug_report_file} in java_src_dict!")
        #     print(f"Sample content:\n{java_src_dict[sample_bug_report_file][:200]}...")
        # else:
        #     print(f"{sample_bug_report_file} is missing in java_src_dict!")

        # # Compare extracted paths from both sources (java_src_dict and bug_reports["files"])
        # bug_report_files = set(file for br in bug_reports for file in br["files"])
        # java_src_files = set(java_src_dict.keys())

        # # Find missing files
        # missing_files = bug_report_files - java_src_files
        # extra_files = java_src_files - bug_report_files

        # print(f"Files in bug_reports but missing in java_src_dict: {len(missing_files)}")
        # print(f"Files in java_src_dict but not in bug_report: {len(extra_files)}")

        # # Print a few missing files
        # print("Example missing files:", list(missing_files)[:5])
        # print("Example extra files: ", list(extra_files)[:5])

        # # Use all CPUs except one to speed up extraction and avoid computer lagging
        # batches = Parallel(n_jobs=cpu_count() - 1) (
        #     delayed(extract)(i, br, bug_reports, java_src_dict)
        #     for i, br in enumerate(bug_reports)
        # )

        # # Flatten features
        # features = [row for batch in batches for row in batch]

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
    
        
