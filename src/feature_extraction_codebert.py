
import os
import time
from utils.util import git_clone, tsv2dict
from feature_extraction import checkout_code_at_commit, get_all_java_files, read_file
from collections import defaultdict
from tqdm import tqdm
import random
import csv



class CodeTimer:
    def __init__(self, name="Code Timer"):
        self.name = name
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        print(f"{self.name} took {elapsed:.2f} seconds")

def prepare_dataset_for_codebert():
    '''
    Prepare dataset for fine-tuning CodeBERT with bug report and source file pairs.

    This method processes bug reports, samples positive and negative pairs, and creates a dataset with bug report,
    source code, and binary labels.

    The final dataset is saved as dataset_codebert.csv
    '''

    # Keep time while preparing dataset
    with CodeTimer("Dataset Preparation for CodeBERT"):

        # Get the current directory (assuming the script is in the src folder)
        current_dir = os.path.dirname(__file__)

        # Navigate up one level from the src folder to reach the parent directory
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

        # Construct the path to the data folder
        data_folder_path = os.path.join(parent_dir, 'data')

        # Define the path to the repository's bundles folder
        repo_dir = os.path.join(data_folder_path, "eclipse.platform.ui")
        repo_bundles_dir = os.path.join(repo_dir, 'bundles')

        # Clone git repo to a local folder
        git_clone(
            repo_url = "https://github.com/eclipse-platform/eclipse.platform.ui.git",
            clone_folder = data_folder_path,
        )

        # Read bug reports from tab-separated file
        file_path = os.path.join(data_folder_path, 'Eclipse_Platform_UI.txt')
        bug_reports = tsv2dict(file_path)

        # Sort bug reports by commit_timestamp for processing in chronological order
        bug_reports.sort(key=lambda br: float(br["commit_timestamp"]) if br["commit_timestamp"] else 0)

        # Group bug reports by commit hash
        commit_groups = defaultdict(list)
        for br in bug_reports:
            commit_sha = br["commit"]
            if not commit_sha:
                print(f"No commit SHA found for bug report ID {br['id']}. Skipping.")
                continue
            commit_groups[commit_sha].append(br)

        all_pairs = []
        skipping_count = 0

        # Process each commit group: checkout the commit once, then process bug reports
        for commit_sha, group in tqdm(commit_groups.items(), desc="Processing Commits", unit="commit", total=len(commit_groups)):
            # Checkout the commit
            if not checkout_code_at_commit(commit_sha, repo_dir):
                print(f"Skipping commit {commit_sha} due to checkout failure.")
                continue

            buggy_files = set()
            for br in group:
                if "files" in br and br["files"]:
                    for file in br["files"]:
                        buggy_files.add(file)

            non_buggy_files = {
                f for f in get_all_java_files(repo_bundles_dir) if f not in buggy_files    
            }

            for br in group:
                # Get bug report text
                br_text = br["raw_text"]
                br_id = br["id"]

                # Create positive pairs
                for file in br["files"]:
                    src = read_file(file)
                    if not src.strip():
                        print(f"Skipping due to missing buggy file {file}.")
                        skipping_count += 1
                        continue
                    all_pairs.append([br_text, file, src, 1]) # Positive pair

                # Create negative pairs (sample 50 non-buggy files)
                non_buggy_sample = random.sample(list(non_buggy_files), min(50, len(non_buggy_files)))
                for file in non_buggy_sample:
                    src = read_file(file)
                    if not src.strip():
                        print(f"Skipping due to missing non-buggy file {file}.")
                        continue
                    all_pairs.append([br_text, file, src, 0])  # Negative pair
        
        # Save dataset to dataset.csv
        save_dataset_to_csv(all_pairs, os.path.join(data_folder_path, "dataset_codebert.csv"))
        print(f"Dataset preparation complete. Dataset saved to dataset_codebert.csv")
        
def save_dataset_to_csv(pairs, path):
    '''
    Save Dataset with bug report, source code, and label to CSV.
    '''
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bug_report", "file name", "source_code", "label"])
        writer.writerows(pairs)


