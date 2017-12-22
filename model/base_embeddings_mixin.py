import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm

__author__ = 'georgi.val.stoyan0v@gmail.com'

class BaseEmbeddingsMixin(object):

    def __init__(self, directory=None, file_name=None):
        self._emb_directory = directory
        self._emb_file_name = file_name
        self._emb_model = None
        self._emb_model_name = None
        self._emb_size = 0

    def vectorize_word(self, word):
        assert self._emb_model, "Embeddings model should be loaded first!"

        return self._emb_model[word]

    def load_embeddings_model(self):
        print("Loading {} model..".format(self._emb_model_name))

        with open(self._emb_directory + self._emb_file_name, 'r') as f:
            model = {}
            split_line = []
            for line in tqdm(f):
                split_line = line.split()
                word = split_line[0]
                embedding = np.array([float(val) for val in split_line[1:]])
                model[word] = embedding

            self._emb_size = len(split_line) - 1

        self._emb_model = model
        print("Done.", len(model), " words loaded!")

    def vectorize_phrases(self, phrases):
        meaning_phrases = []
        meaning_labels = []

        for phrase in tqdm(phrases):
            vectorized_phrase = self.vectorize_phrase(phrase)
            if vectorized_phrase is not None:
                meaning_phrases.append(vectorized_phrase)
                meaning_labels.append(phrase)

        return meaning_phrases, meaning_labels

    def vectorize_phrase(self, string_phrase):
        emb_sum = None
        keys = string_phrase.split()

        all_in_glove = True

        for key in keys:
            if key not in self._emb_model:
                all_in_glove = False
                break

        if all_in_glove:
            emb_sum = np.zeros(self._emb_size)

            for key in keys:
                vec = self._emb_model[key]
                emb_sum += vec

            emb_sum /= len(keys)
            emb_sum = normalize(emb_sum[:, np.newaxis], axis=0).ravel()

        return emb_sum
