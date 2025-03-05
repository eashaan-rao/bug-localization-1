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
from tqdm import tqdm

# Helper functions

class CodeTimer:
    def __init__(self, name="Code Timer"):
        self.name = name
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        print(f"{self.name} took {elapsed:.2f} seconds")

def get_parent_commit(commit_sha, repo_dir):
    '''
    Finds the immediate parent commit of the given commit SHA.

    Args:
        commit_sha (str): The commit hash to find the parent of.
        repo_dir (str): Path to the Git repository.

    Returns:
        str or None: Parent commit hash, or None if not found.
    '''
    cmd =["git", "rev-parse", f"{commit_sha}^"]
    try:
        parent_commit = subprocess.check_output(cmd, cwd=repo_dir).decode("utf-8").strip()
        return parent_commit
    except subprocess.CalledProcessError:
        print(f"Error: Could not find parent commt for {commit_sha}.")
        return None
    
def checkout_code_at_commit(commit_sha, repo_dir):
    '''
    Checks out the repository at the buggy version (one commit before the given SHA)

    Args:
        commit_sha (str): The fix commit hash.
        repo_dir (str) : Path to the Git repository

    Returns:
        bool: True if checkout was successful, False otherwise
    
    '''
    # parent_commit = get_parent_commit(commit_sha, repo_dir)
    # if not parent_commit:
    #     print(f"Skipping {commit_sha} as no parent commit found.")
    #     return False
    
    try:
        # subprocess.run(["git", "checkout", "main"], check=True, cwd=repo_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "checkout", commit_sha], check=True, cwd=repo_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # print(f"Checked out buggy version: {parent_commit} (before fix {commit_sha})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error Checking out {commit_sha}: {e}")
        return False
    
    
def get_all_java_files(repo_bundles_dir):
    '''
        Recursively finds all Java source files in the given repository directory.

        Args:
            repo_bundles_dir (str) : Path to the source code directory

        Returns:
            list: list of absolute paths of Java source files
    '''
    java_files = []
    for root, _, files in os.walk(repo_bundles_dir):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))
    return java_files

def read_file(file_path):
    '''
    Reads the contents of a Java source file.

    Args:
        file_path (str) : Path to the Java file

    Returns:
        str: Content of the file as a string, or an empty string if there's an error 
    '''
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def extract_features_for_bug_report(br, buggy_files, indexed_non_buggy, bug_reports):
    '''
    Extract features for a single bug report
    '''
    features = []
    br_id, br_text = br["id"], br["raw_text"]
    buggy_found = False

    for file in br['files']:
        if file in buggy_files:
                src = read_file(file)
                if not src.strip():
                    print(f"Skipping bug report {br_id}: due to missing file")
                    return []
                features.append(compute_features(bug_reports, int(br_id), file, br_text, src, 1))
                buggy_found = True

    if not buggy_found:
        return []
    
    non_buggy_sample = random.sample(list(indexed_non_buggy.items()), min(50, len(indexed_non_buggy)))
    for file, src in non_buggy_sample:
        features.append(compute_features(bug_reports, int(br_id), file, br_text, src, 0))
    
    return features

def compute_features(bug_reports, br_id, file, br_text, src, label):
    '''
        compute all five features for a bug report source file pair.
    '''

    try:
        bug_report = bug_reports[br_id-1]
        # Ensure valid inputs
        if not isinstance(br_text, str) or not isinstance(src, str):
            raise ValueError("Invalid input: bug report text or source code is not a string")
                
       # rVSM Text Similarity
        rvsm = cosine_sim(br_text, src)

        # Class Name Similarity
        cns = class_name_similarity(br_text, src)

        # Previous Reports
        prev_reports = previous_reports(file, bug_report["report_time"], bug_reports) #change br_text to report_timestamp

        # Collaborative Filter Score
        cfs = collaborative_filtering_score(br_text, prev_reports)

        # Bug Fixing Recency
        bfr = bug_fixing_recency(bug_report, prev_reports)

        # Bug Fixing Frequency
        bff = len(prev_reports)
    
    except Exception as e:
        print(f"Error processing bug report {br_id} for file {file}: {e}")
        return [br_id, file, None, None, None, None, None, label]
            

    return [br_id, file, rvsm, cfs, cns, bfr, bff, label]

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

        # Sort bug reports by commit_timestamp (or report time) so that we can process in chronological order
        bug_reports.sort(key=lambda br: float(br["commit_timestamp"]) if br["commit_timestamp"] else 0)
        # exit(0) b 
        # Group bug reports by the commit hash.
        commit_groups = defaultdict(list)
        for br in bug_reports:
            commit_sha = br["commit"]
            if not commit_sha:
                print(f"No commit SHA found for bug report ID {br['id']}. Skipping")
                continue
            commit_groups[commit_sha].append(br)
        
        # print(f"Grouped bug reports into {len(commit_groups)} commit groups.")

        all_features = []
        indexed_non_buggy = {}
        skipping_count = 0
        # Process each group: checkout the commit once and then extract features for each bug report.
        for commit_sha, group in tqdm(commit_groups.items(), desc="Processin Commits", unit="commit", total=len(commit_groups)):
            # print(f"Processing group for commit {commit_sha} with {len(group)} bug reports.")

            # Get the parent commit to extract the buggy version
            if not checkout_code_at_commit(commit_sha, repo_dir):
                print(f"Skipping commit {commit_sha} due to checkout failures")
                continue

            buggy_files = set()
            for br in group:
                if "files" in br and br["files"]:
                    for file in br["files"]:  
                        buggy_files.add(file)

            non_buggy_files = {
                f for f in get_all_java_files(repo_bundles_dir) if f not in buggy_files
            }
            for f in non_buggy_files:
                    if f not in indexed_non_buggy:
                        indexed_non_buggy[f] = read_file(f)

            for br in group:
                features = extract_features_for_bug_report(br, buggy_files, indexed_non_buggy, bug_reports)
                if not features:  # Skip if no features were extracted
                    print(f"Skipping bug report {br['id']} due to missing files or errors.")
                    skipping_count += 1
                    continue
                all_features.extend(features)
            if all_features:
                save_features_to_csv(all_features, os.path.join(data_folder_path, 'features.csv'))
    print("skipping count: ", skipping_count)
    print(f"Feature extraction complete. Features saved to {os.path.join(data_folder_path, 'features.csv')}")

def save_features_to_csv(features, path):
    '''
    Save features to a features.csv file
    '''
    with open(path, "w", newline="") as f:
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
        writer.writerows(features)


    
        
