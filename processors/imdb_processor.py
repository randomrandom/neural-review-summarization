from collections import Counter

from processors.base_processor import BaseProcessor

__author__ = 'georgi.val.stoyan0v@gmail.com'


class IMDBProcessor(BaseProcessor):
    _DATA_SET_NAME = 'imdb'

    _MIN_UNIGRAMS_COUNT = 30
    _MIN_BIGRAMS_COUNT = 5
    _MIN_TRIGRAMS_COUNT = 2

    def __init__(self, data_file):
        super().__init__(data_file)
        self._data_set_name = self._DATA_SET_NAME

    def _reformat_data(self, data_frame):
        return data_frame

    def _prune_n_grams(self):
        allowed_unigrams = {word: count for word, count in list(Counter(self.unigrams).items()) if
                            count > self._MIN_UNIGRAMS_COUNT}
        allowed_bigrams = {word: count for word, count in list(Counter(self.bigrams).items()) if
                           count > self._MIN_BIGRAMS_COUNT}
        allowed_trigrams = {word: count for word, count in list(Counter(self.trigrams).items()) if
                            count > self._MIN_TRIGRAMS_COUNT}

        filtered_unigrams = [x for x in self.unigrams if x in allowed_unigrams]
        filtered_bigrams = [x for x in self.bigrams if x in allowed_bigrams]
        filtered_trigrams = [x for x in self.trigrams if x in allowed_trigrams]

        return filtered_unigrams, filtered_bigrams, filtered_trigrams

    def _get_data_set_name(self):
        return self._data_set_name
