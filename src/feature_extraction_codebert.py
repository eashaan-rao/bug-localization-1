
import os
import time
from utils.util import git_clone, xlsx2dict
from feature_extraction import checkout_code_at_commit, get_all_java_files, read_file
from collections import defaultdict
from tqdm import tqdm
import random
import csv


PROJECT_REPO_MAPPING = {
    "aspectj": "https://github.com/eclipse-aspectj/aspectj.git",
    "jdt": "https://github.com/eclipse-jdt/eclipse.jdt.ui.git",
    "swt": "https://github.com/eclipse-platform/eclipse.platform.swt.git",
    "birt": "https://github.com/eclipse-birt/birt.git",
    "tomcat": "https://github.com/apache/tomcat.git",
    "eclipse_platform_ui": "https://github.com/eclipse-platform/eclipse.platform.ui.git" 
}

class CodeTimer:
    def __init__(self, name="Code Timer"):
        self.name = name
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        print(f"{self.name} took {elapsed:.2f} seconds")

def prepare_dataset_for_codebert(project_name):
    '''
    Prepare dataset for fine-tuning CodeBERT for a specific project.

    This method processes bug reports, samples positive and negative pairs, and creates a dataset with bug report,
    source code, and binary labels, saved as {project_name}_codebert.csv.

    Args:
        project_name (str): The name of the project to process (e.g., "AspectJ", "JDT").
    '''

    if project_name not in PROJECT_REPO_MAPPING:
        print(f"Error: Project '{project_name}' not found in PROJECT_REPO_MAPPING.")
        return

    # Keep time while preparing dataset
    with CodeTimer("Dataset Preparation for CodeBERT for project: {project_name}"):

        # Get the current directory (assuming the script is in the src folder)
        current_dir = os.path.dirname(__file__)

        # Navigate up one level from the src folder to reach the parent directory
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

        # Construct the path to the data folder
        data_folder_path = os.path.join(parent_dir, 'data')

        # Define paths and URL dynamically based on project_name
        repo_dir = os.path.join(data_folder_path, project_name)
        repo_url = PROJECT_REPO_MAPPING[project_name]

        # Clone git repo to a local folder
        if os.path.exists(repo_dir):
            print(f"Path already exists: {repo_dir}")
            print("Not cloning the repository....")
        else:
            git_clone(
                repo_url = repo_url,
                clone_folder = data_folder_path,
            )

        # Read bug reports from the project's .xlsx file
        xlsx_file_path = os.path.join(parent_dir, 'dataset', f'{project_name}.xlsx')
        bug_reports = xlsx2dict(xlsx_file_path)

        if not bug_reports:
            print(f"No bug reports loaded from {xlsx_file_path}. Aborting...")
            return

        # Sort bug reports by commit_timestamp for processing in chronological order
        bug_reports.sort(key=lambda br: float(br.get("commit_timestamp", 0)) if br.get("commit_timestamp") else 0)

        # Group bug reports by commit hash
        commit_groups = defaultdict(list)
        for br in bug_reports:
            commit_sha = br.get("commit")
            if not commit_sha:
                print(f"No commit SHA found for bug report ID {br.get('id')}. Skipping.")
                continue
            commit_groups[commit_sha].append(br)

        all_pairs = []
        skipping_count = 0

        # Process each commit group: checkout the commit once, then process bug reports
        for commit_sha, group in tqdm(commit_groups.items(), desc=f"Processing Commits for {project_name}", unit="commit"):
            # Checkout the commit
            if not checkout_code_at_commit(commit_sha, repo_dir):
                print(f"Skipping commit {commit_sha} due to checkout failure.")
                continue

            buggy_files_in_group = {file for br in group if br.get("files") for file in br["files"]}
            # Scan for Java files from the repo root for consistency
            all_java_files_in_repo = get_all_java_files(repo_dir)

            non_buggy_files = {
                f for f in all_java_files_in_repo if f not in buggy_files_in_group    
            }

            for br in group:
                # Get bug report text
                br_text = br["raw_text"]

                # Create positive pairs from the confirmed buggy file for this bug report
                buggy_files_for_br = set(br.get("files", []))
                for file_path in buggy_files_for_br:
                    src = read_file(file_path)
                    if not src.strip():
                        print(f"Skipping due to missing content in buggy file: {file_path}.")
                        skipping_count += 1
                        continue
                    all_pairs.append([br_text, file_path, src, 1]) # Positive pair

                # Create negative pairs by sampling from the 50 non-buggy files)
                if non_buggy_files:
                    sample_size = min(50, len(non_buggy_files))
                    non_buggy_sample = random.sample(list(non_buggy_files), sample_size)
                    for file_path in non_buggy_sample:
                        src = read_file(file_path)
                        if src.strip():
                            all_pairs.append([br_text, file_path, src, 0])  # Negative pair
                        
        
        # Save dataset to a project-specific CSV file
        output_filemame = f"{project_name}_codebert_dataset.csv"
        output_path = os.path.join(data_folder_path, output_filemame)
        save_dataset_to_csv(all_pairs, output_path)

        print(f"\n Skipped {skipping_count} files due to missing content.")
        print(f"Dataset preparation complete. Dataset saved to {output_filemame}")
        
def save_dataset_to_csv(pairs, path):
    '''
    Save Dataset with bug report, source code, and label to CSV.
    '''
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bug_report", "file_name", "source_code", "label"])
        writer.writerows(pairs)


