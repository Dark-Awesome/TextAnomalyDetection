'''
Word2 to Vec CNN Module
'''

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

DATA_PATH = ''
data = pd.read_csv(DATA_PATH)


def Word2Vec(data: pd.DataFrame):
    data_X = data['title']
    data_Y = data['tag']

    vocab = CountVectorizer().fit(data_X)
    X_vectors = vocab.transform(data_X)

    return X_vectors , data_Y.values
