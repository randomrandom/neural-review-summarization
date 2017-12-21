import nltk
import pandas as pd
from tqdm import tqdm
from nltk import sent_tokenize
import matplotlib.pyplot as plt
from collections import Counter
from abc import abstractclassmethod

__author__ = 'georgi.val.stoyan0v@gmail.com'


class BaseProcessor(object):
    _DELIM_TSV = '\t'
    _REVIEW_KEY = 'Review'
    _PHRASE_KEY = 'Phrase'
    _COUNT_KEY = 'Count'
    _REGEX_FILTER = r'\p{P}+'
    _ALLOWED_TAGS = {'JJ', 'JJ NN', 'JJ NNS', 'JJ NN NN', 'RB JJ NN', 'JJ TO VB', 'VB JJ NN'}
    _BANNED_WORDS = {'', 'br'}

    _OUTPUT_DIRECTORY = '/mnt/c/Users/gvs/ubuntu/neural-review-summarization/output/'
    _ALL_GRAMS_FILE = 'all_grams.csv'
    _UNIGRAMS_FILE = 'unigrams.csv'
    _BIGRAMS_FILE = 'bigrams.csv'
    _TRIGRAMS_FILE = 'trigrams.csv'

    def __init__(self, data_file):
        self._data_file = data_file
        self._data_frame = None
        self.unigrams = []
        self.bigrams = []
        self.trigrams = []
        self.all_grams = []

    def load_data(self, sep=_DELIM_TSV):
        df = pd.read_csv(self._data_file, sep=sep)
        proper_df = self._reformat_data(df)
        self._data_frame = proper_df

        return self._data_frame

    @abstractclassmethod
    def _reformat_data(self, data_frame):
        raise NotImplementedError

    @abstractclassmethod
    def _get_data_set_name(self):
        raise NotImplementedError

    def prepare_data(self):
        print('Generating n_grams')
        for i in tqdm(range(len(self._data_frame))):
            words, tags = self._generate_words_and_tags(self._data_frame.iloc[i][self._REVIEW_KEY])

            self.unigrams.extend(self._generate_n_grams(words, tags, 1)[0])
            self.bigrams.extend(self._generate_n_grams(words, tags, 2)[0])
            self.trigrams.extend(self._generate_n_grams(words, tags, 3)[0])

        print('Filtering n_grams')
        self.unigrams, self.bigrams, self.trigrams = self._prune_n_grams()

        self.all_grams.extend(self.unigrams)
        self.all_grams.extend(self.bigrams)
        self.all_grams.extend(self.trigrams)
        print('Done.')

    def save_n_grams(self):
        n_gram_counts = [Counter(self.all_grams), Counter(self.unigrams), Counter(self.bigrams), Counter(self.trigrams)]
        n_gram_files = [self._ALL_GRAMS_FILE, self._UNIGRAMS_FILE, self._BIGRAMS_FILE, self._TRIGRAMS_FILE]

        for i in range(len(n_gram_files)):
            n_gram_file = n_gram_files[i]
            n_gram_count = n_gram_counts[i]

            n_gram_df = pd.DataFrame.from_dict(n_gram_count, orient='index').reset_index()
            n_gram_df = n_gram_df.rename(columns={'index': 'Phrase', 0: 'Count'})
            n_gram_df.to_csv(self._OUTPUT_DIRECTORY + n_gram_file, encoding='utf-8')

        print('Saved n_gram files!')

    def load_n_grams(self):
        all_gram_df = pd.read_csv(self._OUTPUT_DIRECTORY + self._ALL_GRAMS_FILE)
        unigram_df = pd.read_csv(self._OUTPUT_DIRECTORY + self._UNIGRAMS_FILE)
        bigram_df = pd.read_csv(self._OUTPUT_DIRECTORY + self._BIGRAMS_FILE)
        trigram_df = pd.read_csv(self._OUTPUT_DIRECTORY + self._TRIGRAMS_FILE)

        # TODO: we are loosing the count distributions after loading the model
        # the items are a set rather than a list with repeating items as they
        # used to be before saving the model
        all_grams_counts = self._df_to_dict(all_gram_df)
        self.all_grams = [n_gram for n_gram, count in all_grams_counts.items()]
        unigrams_counts = self._df_to_dict(unigram_df)
        self.unigrams = [n_gram for n_gram, count in unigrams_counts.items()]
        bigrams_counts = self._df_to_dict(bigram_df)
        self.bigrams = [n_gram for n_gram, count in bigrams_counts.items()]
        trigrams_counts = self._df_to_dict(trigram_df)
        self.trigrams = [n_gram for n_gram, count in trigrams_counts.items()]

        return all_gram_df, unigram_df, bigram_df, trigram_df

    def _df_to_dict(self, df):
        count_dict = {}

        for i in range(len(df)):
            string_phrase = df.iloc[i][self._PHRASE_KEY]
            phrase_count = df.iloc[i][self._COUNT_KEY]

            count_dict[string_phrase] = phrase_count

        return count_dict

    @staticmethod
    def print_most_common(n_grams, most_common=100):
        counts = Counter(n_grams)
        print(counts.most_common(most_common))

    @staticmethod
    def print_lest_common(n_grams, least_common=100):
        counts = Counter(n_grams)
        print(counts.most_common()[-least_common:])

    @abstractclassmethod
    def _prune_n_grams(self):
        raise NotImplementedError

    @staticmethod
    def _filter_banned_words(words, banned_words):
        filtered_words = list(filter(lambda x: x not in banned_words, words))

        return filtered_words

    @staticmethod
    def _merge_n_grams(n_grams, n_tags):
        merged_n_grams = []
        merged_n_tags = []

        for n_gram, n_tag in zip(n_grams, n_tags):
            merged_n_grams.append(' '.join(n_gram))
            merged_n_tags.append(' '.join(n_tag))

        return merged_n_grams, merged_n_tags

    @staticmethod
    def _filter_by_tags(n_grams, n_tags, allowed_tags):
        pos = zip(n_grams, n_tags)

        filtered_pos = list(filter(lambda x: len(x[0]) > 1 and x[1] in allowed_tags, pos))

        # filtered tags and words
        filtered_words = [word for word, tag in filtered_pos]
        filtered_tags = [tag for word, tag in filtered_pos]

        return filtered_words, filtered_tags

    @staticmethod
    def visualize_n_grams_distribution(n_grams):
        n_gram_counts = [x for x in list(Counter(n_grams).values())]
        print('Total n_grams {}'.format(len(n_gram_counts)))
        plt.hist(n_gram_counts, bins=300)
        plt.ylabel('items')
        plt.xlabel('n_gram counts')

    def _generate_words_and_tags(self, review):
        sentences = sent_tokenize(review)

        # pos tagging and filtering
        pos_tags = []
        for sentence in sentences:
            split_words = [nltk.re.sub(self._REGEX_FILTER, '', x).lower() for x in sentence.split()]
            words = self._filter_banned_words(split_words, self._BANNED_WORDS)
            pos_tags.extend(nltk.pos_tag(words))

        # split words and tags
        words = [word.lower() for word, tag in pos_tags]
        tags = [tag for word, tag in pos_tags]

        return words, tags

    def _generate_n_grams(self, words, tags, n=1):
        n_grams = None
        n_tags = None

        if n == 1:
            n_grams, n_tags = self._filter_by_tags(words, tags, self._ALLOWED_TAGS)
        elif n == 2:
            bigrams = zip(words, words[1:])
            bitags = zip(tags, tags[1:])

            merged_bigrams, merged_bitags = self._merge_n_grams(bigrams, bitags)
            n_grams, n_tags = self._filter_by_tags(merged_bigrams, merged_bitags, self._ALLOWED_TAGS)
        elif n == 3:
            trigrams = zip(words, words[1:], words[2:])
            tritags = zip(tags, tags[1:], tags[2:])

            merged_trigrams, merged_tritags = self._merge_n_grams(trigrams, tritags)
            n_grams, n_tags = self._filter_by_tags(merged_trigrams, merged_tritags, self._ALLOWED_TAGS)

        return n_grams, n_tags
