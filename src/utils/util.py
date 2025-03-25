'''
Helper functions for feature extraction from the bug localization dataset
'''

import csv
import re
import os
import random
import timeit
import string
import nltk
import numpy as np
import pandas as pd
from datetime import datetime
import subprocess
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
# Cache stopwords and stemmer
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

def git_clone(repo_url, clone_folder):
    '''
    Clones the git repo from 'repo_url' into 'clone_folder'

    Arguments:
    repo_url {string} -- url of git repository
    clone_folder {string} -- path of a local folder to clone the repository
    '''
    repo_name = os.path.basename(repo_url)
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    target_path = os.path.join(clone_folder, repo_name)
    if os.path.isdir(target_path):
        print("Already cloned")
        return
    
    # Ensure the clone folder exists
    if not os.path.isdir(clone_folder):
        try:
            os.mkdir(clone_folder)
        except OSError as e:
            print(f"Error creating directory {clone_folder}: {e}")
    
    # Clone the repository
    try:
        subprocess.run(["git", "clone", repo_url, target_path], check=True)
        print(f"Cloned {repo_url} into {target_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while cloning the repository: {e}")


def tsv2dict(tsv_path):
    '''
    Converts a tab separated values (tsv) file into a list of dictionaries

    Arguments:
    tsv_path {string} -- path of the tsv file
    '''
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    data_folder_path = os.path.join(parent_dir, 'data')
    repo_dir = os.path.join(data_folder_path, "eclipse.platform.ui")
    repo_bundles_dir = os.path.join(repo_dir, 'bundles')
    with open(tsv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        dict_list = []
        for line in reader:
            # Extract valid Java files, removing prefixes like "6:"
            
            if line["files"]:
                processed_files = []
                 # Regex pattern to match valid file paths ending with .java
                # It looks for paths starting with bundles/, examples/, or test/
                pattern = r'\b(?:bundles|examples|test)\/.*?\.java'

                # Find all matching file paths
                matches = re.findall(pattern, line["files"])
                # print("matches: ", matches)

                for f in matches:
                    # print("f: ", f)
                    # f = f.strip()
                    full_path = os.path.normpath(os.path.join(repo_dir, f))
                    processed_files.append(full_path)

                # for f in line["files"].split(" "):
                #     if f.strip():
                #         f = f.split(":", 1)[-1]  # Remove any prefix like "6:"
                #         f = "bundles/" + f.strip()
                #         if f.endswith(".java"):
                #             full_path = os.path.normpath(os.path.join(repo_dir, f))
                #             processed_files.append(full_path)
                line['files'] = processed_files
            else:
                line["files"] = []

            # combine summary and description safely
            line["raw_text"] = " ".join([line["summary"].strip() , line["description"].strip()])

            # Conver report time, handling missing values
            line["report_time"] = (
                datetime.strptime(line["report_time"], "%Y-%m-%d %H:%M:%S")
                if line["report_time"]
                else None
            )

            # Keep only required keys
            filtered_line = {
                key: line[key] for key in reader.fieldnames if key == "files" or key in ["id", "bug_id", "summary", "description", "report_time", "status", "commit", "commit_timestamp"]
            }
            filtered_line["raw_text"] = line["raw_text"]

            dict_list.append(filtered_line)
    
    return dict_list

def csv2dict(csv_path):
    '''
    Converts a comma separated values (csv) file into a dictionary
    
    Arguments:
    csv_path {string} -- path to csv file
    '''

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter=",")
        csv_dict = list()
        for line in reader:
            csv_dict.append(line)

    return csv_dict

def clean_text(text):
    '''
    Lowercase, remove punctuation, and normalize whitespace

    Arguments:
    text {string} -- input text
    '''

    text = text.lower()
    text = text.translate(str.maketrans('','', string.punctuation)) # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    
    return text

def top_k_wrong_files(right_files, br_raw_text, java_files, k=50):
    '''
    Randomly samples 2*k from all wrong files and returns metrics for top k files according to rvsm similarity
    
    Arguments:
        right_files {list} -- list of right files
        br_raw_text {string} -- raw text of the bug report
        java_files {dictionary} -- dictionary of source code files
    
    Keyword Arguments:
        k {integer} -- the number of files to return metrics (default: {50})
    '''

    # Randomly sample 2*k files
    randomly_sampled = random.sample(set(java_files.keys()) - set(right_files), 2*k)

    all_files = []
    for filename in randomly_sampled:
        try:
            src = java_files[filename]
            rvsm = cosine_sim(br_raw_text, src)
            cns = class_name_similarity(br_raw_text, src)

            all_files.append((filename, rvsm, cns))
        except:
            pass

    top_k_files = sorted(all_files, key=lambda x: x[1], reverse=True)[:k]

    return top_k_files

def tokenize_and_stem(text):
    '''
        Tokenize, remove stopwords, and stem. (combining stem_tokens() and normalize)

        Arguments:
            text {string} -- Preprocessed Text
    '''
    tokens = word_tokenize(text)
    stemmed_tokens = [STEMMER.stem(token) for token in tokens if token.lower() not in STOP_WORDS]
    return stemmed_tokens


def cosine_sim(text1, text2):
    '''
    Cosine similarity with tfidf

    Arguments:
        text1 {string} -- first text
        text2 {string} -- second text
    '''
    vectorizer = TfidfVectorizer(preprocessor=clean_text, tokenizer=tokenize_and_stem, token_pattern=None)
    tfidf = vectorizer.fit_transform([text1, text2])
    sim = ((tfidf * tfidf.T).toarray())[0,1]

    return sim

def get_relevant_source_code(relevant_files, base_dir):
    '''
    Extract only the Java source files that are referenced in the bug report.

    Arguments:
        relevant_files {list} -- List of relative file paths from the bug report
        base_dir {string} -- Path to the repository's bundles directory (the checked-out version).
    Returns:
        dict -- A dictionary mapping file paths to their content
    '''

    source_dict = {}
    for file in relevant_files:
        # Construct full path - make sure the file path format matches what is in the repo
        file_path = os.path.join(base_dir, file)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    source_dict[file] = f.read()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        else:
            #print missing files 
            print(f"File not found in repo: {file_path}")
    return source_dict

def get_all_source_code(start_dir):
    '''
    Creates corpus starting from 'start_dir'

    Arguments:
        start_dir {string} -- directory path to start
    '''
    files = {}
    start_dir = os.path.normpath(start_dir)

    for dir_, _, file_names in os.walk(start_dir):
        for filename in file_names:
            if not filename.endswith(".java"):
                continue

            src_name = os.path.join(dir_, filename)

            # Read file safely
            try:
                with open(src_name, "r", encoding="utf-8", errors="ignore") as src_file:
                    src = src_file.read()
            except Exception as e:
                print(f"Error reading {src_name}: {e}")
                continue # skip problematic files

            # Create file key (relative path without "bundles/")         
            file_key = os.path.relpath(src_name, start_dir)
            if file_key.startswith("bundles/"):
                file_key = file_key[8:]

            files[file_key] = src 
    return files 

def get_months_between(d1, d2):
    '''
    Calculates the number of months betwween two date strings

    Arguments:
        d1 {datetime} -- date 1
        d2 {datetime} -- date 2
    '''  

    diff_in_months = abs((d1.year - d2.year) * 12 + d1.month - d2.month)

    return diff_in_months

def most_recent_report(reports):
    '''
    Returns the most recently submitted previous report that shares a filename with the given bug report

    Arguments:
        reports {} --  find it's meaning! (remark)
    '''

    if len(reports) > 0:
        return max(reports, key=lambda x: x.get("report_time"))
    
    return None

def previous_reports(filename, until, bug_reports):
    '''
    Returns a list of previously filed bug reports that share a file with the current bug report

    Arguments:
        filename {string} -- the name of the shared Java file
        current_date {datetime} -- until date
        bug_reports {list of dictionaries} -- list of all bug reports
    '''
    return [
        br 
        for br in bug_reports
        if (filename in br["files"] and br["report_time"] and br["report_time"] < until)
    ]

def bug_fixing_recency(br, prev_reports):
    '''
    Calculates the bug fixing recency as defined by Lam et al.

    Arguments:
        br {} -- bug report
        prev_reports {} -- previous bug reports (find the data types)
    '''
    if "report_time" not in br:
        return 0 # Default value is missing

    mrr = most_recent_report(prev_reports)

    if mrr and "report_time" in mrr:
        return 1 / float (
            get_months_between(br["report_time"], mrr["report_time"]) + 1
        )
    
    return 0

def collaborative_filtering_score(raw_text, prev_reports):
    '''
    Calculates ...

    Arguments:
        raw_text {string} -- raw text of the bug report
        prev_reports {list} -- list of previous reports
    '''
    if not prev_reports:
        return 0 # No previous reports - No similarity

    prev_reports_merged_raw_text = "".join(report["raw_text"] for report in prev_reports if "raw_text" in report)
    cfs = cosine_sim(raw_text, prev_reports_merged_raw_text)

    return cfs

def class_name_similarity(raw_text, source_code):
    '''
    Calculates the class name present from bug reports and source code 
    
    Arguments:
        raw_text {string} -- raw text of the bug report
        source_code {string} -- java source code
    '''

    class_names = re.findall(r'class\s+(\w+)', source_code)
    if not class_names:
        return 0 # No class found
    
    class_names_text = " ".join(class_names)

    return cosine_sim(raw_text, class_names_text)


class CodeTimer:
    '''
    Keeps time from the initialization, and print the elapsed time at the end.
    '''

    def __init__(self, message=""):
        self.message = message
    
    def __enter__(self):
        print(self.message)
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = timeit.default_timer() - self.start
        print("Finished in {0:0.5f} secs".format(self.took))
        