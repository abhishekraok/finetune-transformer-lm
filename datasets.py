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
    trX1 = train_sentence[:train_count]
    trY = train_y[:train_count]
    vaX1 = train_sentence[train_count:]
    vaY = train_y[train_count:]
    return (trX1, trY), (vaX1, vaY), (test_sentence, test_y)
