'''
Helper functions mostly for feature extraction
'''

import csv
import re
import os
import random
import timeit
import string
import numpy as np
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def git_clone(repo_url, clone_folder):
    '''
    Clones the git repo from 'repo_url' into 'clone_folder'

    Arguments:
    repo_url {string} -- url of git repository
    clone_folder {string} -- path of a local folder to clone the repository
    '''
    repo_name = repo_url[repo_url.rfind("/") + 1 : -4]
    if os.path.isdir(clone_folder + repo_name):
        print("Already cloned")
        return
    
    cwd = os.getcwd()
    if not os.path.isdir(clone_folder):
        os.mkdir(clone_folder)
    
    os.chdir(clone_folder)
    os.system("git clone {}".format(repo_url))
    os.chdir(cwd)

def tsv2dict(tsv_path):
    '''
    Converts a tab separated values (tsv) file into a list of dictionaries

    Arguments:
    tsv_path {string} -- path of the tsv file
    '''

    reader = csv.DictReader(open(tsv_path, "r"), delimiter="\t")
    dict_list = []
    for line in reader:
        line["files"] = [
            os.path.normpath(f[8:])
            for f in line["files"].strip().split()
            if f.startswith("bundles/") and f.endswith(".java")
        ]
        line["raw_text"] = line["summary"] + line["description"]
        line["report_time"] = datetime.strptime(
            line["report_time"], "%Y-%m-%d %H:%M:%S"
        )

        dict_list.append(line)
    
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

def clean_and_split(text):
    '''
    Remove all punctuation and split text strings into lists of words

    Arguments:
    text {string} -- input text
    '''

    table = str.maketrans(dict.fromkeys(string.punctuation))
    clean_text = text.translate(table)
    word_list = [s.strip() for s in clean_text.strip().split()]
    return word_list

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
    