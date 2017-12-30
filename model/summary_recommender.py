from scipy import spatial, math

from model.affinity_cluster_model import AffinityClusterModel
from processors.imdb_processor import IMDBProcessor

__author__ = 'georgi.val.stoyan0v@gmail.com'


class SummaryRecommender:
    _PHRASE_KEY = 'phrase'
    _COUNT_KEY = 'count'
    _SCORE_KEY = 'score'
    _N_GRAM_SIZE = 'n_size'

    _SIMILARITY_THRESHOLD = 0.80

    _BAD_SUBSTRINGS = {
        "i",
        "thing",
        "entire",
        "whole",
        "several",
        "previous",
        "other",
        "main",
        "many movie",
        "few movies",
        "new film",
        "much time",
        "pretty much"
    }

    def __init__(self, data_file, encoding='latin-1'):
        self._processor = None
        self._unigrams_cl = None
        self._bigrams_cl = None
        self._trigrams_cl = None

        self.generate_phrases(data_file, encoding=encoding)
        self._init_models()
        self._load_models()

    def generate_phrases(self, data_file, encoding='latin-1'):
        self._processor = IMDBProcessor(data_file)
        self._processor.load_data(encoding=encoding)
        self._processor.prepare_data(test=True)

    def _init_models(self):
        print('Initializing models..')
        self._unigrams_cl = AffinityClusterModel()
        self._bigrams_cl = AffinityClusterModel()
        self._trigrams_cl = AffinityClusterModel()
        print('Initialization of models complete.')

    def _load_models(self):
        print('Loading n-gram models..')
        self._unigrams_cl.load_model(AffinityClusterModel.UNIGRAM_MODEL)
        self._bigrams_cl.load_model(AffinityClusterModel.BIGRAM_MODEL)
        self._trigrams_cl.load_model(AffinityClusterModel.TRIGRAM_MODEL)
        print('Loading of n-gram models completed.')

    def __pack_and_prune_n_grams(self, model, original_to_closest_phrases, n_gram_usage_dict, n_gram_count,
                                 prune_threshold=100):
        packed_n_grams = []

        n_gram_usage = [(k, v) for k, v in n_gram_usage_dict.items()]
        n_gram_usage.sort(key=lambda x: x[1], reverse=True)
        pruned_n_gram_usage = n_gram_usage[:prune_threshold]

        for key, count in pruned_n_gram_usage:
            closest_phrase = original_to_closest_phrases[key]

            packed_n_grams.append({
                self._PHRASE_KEY: key,
                self._COUNT_KEY: count,
                self._SCORE_KEY: model.phrases_to_scores[closest_phrase],
                self._N_GRAM_SIZE: n_gram_count,
            })

        return packed_n_grams

    def _predict_good_phrases(self):
        unigram_predictions, unigram_original_phrases, unigram_closest_phrases = self._unigrams_cl.predict(
            self._processor.unigrams)
        original_to_closest_phrases = dict(zip(unigram_original_phrases, unigram_closest_phrases))
        unigram_phrases_usage = self._unigrams_cl.get_phrases_in_good_clusters(unigram_predictions,
                                                                               unigram_original_phrases)

        bigram_predictions, bigram_original_phrases, bigram_closest_phrases = self._bigrams_cl.predict(
            self._processor.bigrams)
        bigram_phrases_usage = self._bigrams_cl.get_phrases_in_good_clusters(bigram_predictions,
                                                                             bigram_original_phrases)
        original_to_closest_phrases.update(zip(bigram_original_phrases, bigram_closest_phrases))

        trigram_predictions, trigram_original_phrases, trigram_closest_phrases = self._trigrams_cl.predict(
            self._processor.trigrams)
        trigram_phrases_usage = self._trigrams_cl.get_phrases_in_good_clusters(trigram_predictions,
                                                                               trigram_original_phrases)
        original_to_closest_phrases.update(zip(trigram_original_phrases, trigram_closest_phrases))

        all_good_phrases = []
        all_good_phrases.extend(self.__pack_and_prune_n_grams(self._unigrams_cl, original_to_closest_phrases,
                                                              unigram_phrases_usage, 1, prune_threshold=80))
        all_good_phrases.extend(self.__pack_and_prune_n_grams(self._bigrams_cl, original_to_closest_phrases,
                                                              bigram_phrases_usage, 2, prune_threshold=150))
        all_good_phrases.extend(self.__pack_and_prune_n_grams(self._trigrams_cl, original_to_closest_phrases,
                                                              trigram_phrases_usage, 3, prune_threshold=150))

        return all_good_phrases

    def _compress_similar_phrases(self, phrases_to_compress):
        compressed_good_phrases = []

        for i in range(len(phrases_to_compress)):
            first_phrase = phrases_to_compress[i][self._PHRASE_KEY]

            similar_exists = False

            for j in range(len(compressed_good_phrases)):
                second_phrase = compressed_good_phrases[j][self._PHRASE_KEY]
                cosine_similarity = 1 - spatial.distance.cosine(self._unigrams_cl.vectorize_phrase(first_phrase),
                                                                self._unigrams_cl.vectorize_phrase(second_phrase))

                if len(second_phrase.split(' ')) == 1 and second_phrase in first_phrase:
                    similar_exists = True
                    compressed_good_phrases[j][self._COUNT_KEY] += phrases_to_compress[i][self._COUNT_KEY]
                elif cosine_similarity > self._SIMILARITY_THRESHOLD:
                    similar_exists = True
                    compressed_good_phrases[j][self._COUNT_KEY] += phrases_to_compress[i][self._COUNT_KEY]

            if not similar_exists:
                compressed_good_phrases.append(phrases_to_compress[i])

        return compressed_good_phrases

    def _filter_phrases(self, phrases_to_filter):
        compressed_phrases = self._compress_similar_phrases(phrases_to_filter)
        filtered_phrases = []

        bad_substrings = list(self._BAD_SUBSTRINGS)

        for phrase in compressed_phrases:
            bad_substring_found = False

            for bad_substring in bad_substrings:
                if (' ' + bad_substring in phrase[self._PHRASE_KEY]) or (
                                bad_substring + ' ' in phrase[self._PHRASE_KEY]) or (
                                len(bad_substring) > 2 and bad_substring in phrase[self._PHRASE_KEY]):
                    bad_substring_found = True
                    break

            if not bad_substring_found:
                filtered_phrases.append(phrase)

        return filtered_phrases

    def _update_scores(self, phrases_to_update):
        for phrase in phrases_to_update:
            score = phrase[self._COUNT_KEY] * (1 + 0.4 * phrase[self._SCORE_KEY])
            phrase[self._SCORE_KEY] = math.ceil(score * 100) / 100

        return phrases_to_update

    def recommend_phrases(self, recommendation_size=10, unigrams_ratio=.2, bigrams_ratio=.4, trigrams_ratio=.4):
        good_phrases = self._predict_good_phrases()
        filtered_phrases = self._filter_phrases(good_phrases)
        compressed_phrases = self._compress_similar_phrases(filtered_phrases)
        updated_phrases = self._update_scores(compressed_phrases)

        # TODO: add scoring of the phrases

        # separating phrases
        unigram_phrases = list(filter(lambda x: x[self._N_GRAM_SIZE] == 1, updated_phrases))
        bigram_phrases = list(filter(lambda x: x[self._N_GRAM_SIZE] == 2, updated_phrases))
        trigram_phrases = list(filter(lambda x: x[self._N_GRAM_SIZE] == 3, updated_phrases))

        # sorting lists
        unigram_phrases.sort(key=lambda item: item[self._SCORE_KEY], reverse=True)
        bigram_phrases.sort(key=lambda item: item[self._SCORE_KEY], reverse=True)
        trigram_phrases.sort(key=lambda item: item[self._SCORE_KEY], reverse=True)

        trigrams_count = trigrams_ratio * recommendation_size
        bigrams_count = bigrams_ratio * recommendation_size
        recommended_phrases = []

        # fill the wanted ratios from trigrams, bigrams to unigrams
        for phrase in trigram_phrases:
            if phrase[self._COUNT_KEY] >= 2 and len(recommended_phrases) < trigrams_count:
                recommended_phrases.append(phrase)

        for phrase in bigram_phrases:
            if phrase[self._COUNT_KEY] >= 2 and len(recommended_phrases) < (trigrams_count + bigrams_count):
                recommended_phrases.append(phrase)

        for phrase in unigram_phrases:
            if phrase[self._COUNT_KEY] >= 2 and len(recommended_phrases) < recommendation_size:
                recommended_phrases.append(phrase)

        return recommended_phrases
