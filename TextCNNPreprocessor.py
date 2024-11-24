import numpy as np
import pandas as pd
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')

class TextCNNPreprocessor:
    def __init__(self, embedding_path, embedding_dim=300):
        self.embedding_path = embedding_path
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.train_embedding_weights = None

    def load_embeddings(self):
        print("Loading embeddings...")
        embeddings_index = {}
        with open(self.embedding_path, encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        print(f"{len(embeddings_index)} word vectors loaded.")
        return embeddings_index

    def tokenize_and_pad(self, texts, max_words=None, max_sentence_length=None):
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=max_words, char_level=False)
            self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        word_index = self.tokenizer.word_index
        print(f"Vocabulary size: {len(word_index)}")
        padded = pad_sequences(sequences, maxlen=max_sentence_length, padding='post')
        return padded, word_index

    def prepare_embedding_matrix(self, word_index, embeddings_index):
        print("Preparing embedding matrix...")
        num_words = len(word_index) + 1
        embedding_matrix = np.zeros((num_words, self.embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        self.train_embedding_weights = embedding_matrix
        print("Embedding matrix ready.")
        return embedding_matrix

    def create_2d_matrix(self, sequences, m, n):
        def pad_matrix(matrix, m, n):
            extra_rows = max(0, m - len(matrix))
            padded_matrix = pad_sequences(matrix, maxlen=n, padding='post')
            zero_rows = np.zeros((extra_rows, n))
            return np.vstack([padded_matrix, zero_rows])
        
        return np.array([pad_matrix(seq, m, n) for seq in sequences])

    def preprocess(self, texts, max_words=None, max_sentence_length=None, m=10, n=15):
        # Step 1: Tokenize and Pad
        padded, word_index = self.tokenize_and_pad(texts, max_words, max_sentence_length)
        # Step 2: Convert to 2D Matrices
        matrices = self.create_2d_matrix(padded, m, n)
        return matrices, word_index


'''
USAGE EXAMPLE

from text_cnn_preprocessor import TextCNNPreprocessor

# Load your dataset
data_train = pd.read_csv("train.csv")  # Assume it has a 'text' and 'label' column
data_test = pd.read_csv("test.csv")

texts_train = data_train['text'].values
texts_test = data_test['text'].values
labels_train = data_train['label'].values

# Initialize the Preprocessor
embedding_path = "../input/glove6b300dtxt/glove.6B.300d.txt"  # Update with your file path
preprocessor = TextCNNPreprocessor(embedding_path)

# Load embeddings
embeddings_index = preprocessor.load_embeddings()

# Preprocess text
train_matrices, word_index = preprocessor.preprocess(texts_train, max_words=5000, max_sentence_length=100, m=10, n=15)

# Prepare embedding matrix
embedding_matrix = preprocessor.prepare_embedding_matrix(word_index, embeddings_index)

# Your train_matrices are now ready for CNN input
print(train_matrices.shape)  # (num_samples, m, n)

'''