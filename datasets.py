import os
import csv
import numpy as np

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445


def get_sentence(path):
    with open(path, encoding='utf-8') as f:
        y = []
        sentence = []
        for line in f:
            elements = line.split('\t')
            sentence.append(elements[0])
            y.append(int(elements[1]))
        return sentence, y


def extract_sentence(train_filename, test_filename):
    train_sentence, train_y = get_sentence(train_filename)
    test_sentence, test_y = get_sentence(test_filename)
    train_count = int(0.9 * len(train_sentence))
    train_X = train_sentence[:train_count]
    train_Y = train_y[:train_count]
    val_X = train_sentence[train_count:]
    val_Y = train_y[train_count:]
    return (train_X, train_Y), (val_X, val_Y), (test_sentence, test_y)
