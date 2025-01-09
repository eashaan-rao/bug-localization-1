from util import csv2dict, tsv2dict, helper_collections, topK_accuracy
from sklearn.neural_network import MLPRegressor
from joblib import Parallel, delayed, cpu_count
from math import ceil
import numpy as np
import os

def oversample(samples):
    '''
    Oversamples the features for label '1'

    Arguments:
        samples {list} -- samples from features.csv

    '''
    samples_ = []

    # oversample features of buggy files (note: didn't understand the logic)
    for i, sample in enumerate(samples):
        samples_.append(sample)
        if i % 51 == 0:
            for _ in range(9):
                samples_.append(sample)
    
    return samples_

def features_and_labels(samples):
    '''
    Returns features and labels for the given list of samples

    Arguments:
        samples {list} -- samples from features.csv
    
    '''

    features = np.zeros((len(samples), 5))
    labels = np.zeros((len(samples), 1))

    for i, sample in enumerate(samples):
        features[i][0] = float(sample['rVSM_similarity'])
        features[i][1] = float(sample["collab_filter"])
        features[i][2] = float(sample["classname_similarity"])
        features[i][3] = float(sample["bug_recency"])
        features[i][4] = float(sample["bug_frequency"])
        labels[i] = float(sample["match"])

    return features, labels

def kfold_split_indexes(k, len_samples):
    '''
    Returns list of tuples for split start(inclusive) and finish(exclusive) indexes.

    Arguments:
        k {integer} -- the number of folds
        len_samples {integer} -- the length of the sample list
    '''

    step = ceil(len_samples / k)
    ret_list = [(start, start + step) for start in range(0, len_samples, step)]

    return ret_list

def kfold_split(bug_reports, samples, start, finish):
    '''
    Returms train samples and bug reports for test

    Arguments:
        bug_reports {list of dictionaries} -- list of all bug reports
        samples {list} -- samples from features.csv
        start {integer} -- start index from test fold
        finish {integer} -- start index for test fold
    '''

    train_samples = samples[:start] + samples[finish:]
    test_samples = samples[start:finish]

    test_br_ids = set([s["report_id"] for s in test_samples])
    test_bug_reports = [br for br in bug_reports if br["id"] in test_br_ids]

    return train_samples, test_bug_reports