'''
A Script for feature extraction
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