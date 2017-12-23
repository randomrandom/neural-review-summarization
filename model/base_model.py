import pickle
from collections import Counter
from abc import abstractclassmethod
from sklearn.metrics.pairwise import cosine_distances

from model.base_embeddings_mixin import BaseEmbeddingsMixin

__author__ = 'georgi.val.stoyan0v@gmail.com'


class BaseModel(BaseEmbeddingsMixin):
    def __init__(self):
        super().__init__()

        self.n_gram_meanings = None
        self.n_gram_meaning_labels = None
        self.n_gram_meaning_matrix = None

        self._model = self._init_model()

    @abstractclassmethod
    def train_model(self, n_grams, model_name):
        raise NotImplementedError

    @abstractclassmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractclassmethod
    def save_model(self, save_dir, model_name):
        raise NotImplementedError

    @abstractclassmethod
    def load_model(self, model_name):
        raise NotImplementedError

    @staticmethod
    @abstractclassmethod
    def _init_model(self):
        raise NotImplementedError

    @abstractclassmethod
    def get_good_clusters(self):
        raise NotImplementedError

    def cluster_labels(self):
        return self._model.labels_

    def clusters_count(self):
        return len(list(set(self.cluster_labels())))

    def print_cluster_info(self):
        print('Number of clusters: {}'.format(self.clusters_count()))

    def generate_cluster_examples(self):
        phrases = self.n_gram_meaning_labels
        cluster_labels = self.cluster_labels()

        cluster_examples = {}

        for i in range(len(cluster_labels)):
            if cluster_labels[i] not in cluster_examples:
                cluster_examples[cluster_labels[i]] = []

            cluster_examples[cluster_labels[i]].append(phrases[i])

        return cluster_examples

    def print_clusters_examples(self, most_common=50, show=30):
        cluster_examples = self.generate_cluster_examples()
        cluster_counts = Counter(self.cluster_labels())

        for pair in cluster_counts.most_common(most_common):
            cluster_label, _ = pair

            print('\nPrinting examples for cluster {}:'.format(cluster_label))

            size = min(show, len(cluster_examples[cluster_label]))
            for i in range(size):
                print(cluster_examples[cluster_label][i])

    def _generate_phrases_meanings(self, phrases):
        deduped_phrases = list(set(phrases))
        meanings, meaning_labels = self.vectorize_phrases(deduped_phrases)

        return meanings, meaning_labels

    def _generate_n_gram_meanings(self, n_grams):
        print('Vectorizing phrases..')
        self.n_gram_meanings, self.n_gram_meaning_labels = self._generate_phrases_meanings(n_grams)
        print('Vectorization done.')

    def _generate_meaning_matrices(self):
        self.n_gram_meaning_matrix = 1 - cosine_distances(self.n_gram_meanings)

    @staticmethod
    def _load_scikit_model(directory, model_file):
        print('Loading scikit model..')

        with open(directory + model_file, 'rb') as model_file:
            model = pickle.load(model_file)

        print('Scikit model loaded.')

        return model

    @staticmethod
    def _save_scikit_model(model, directory, model_file):
        print('Saving scikit model...')

        with open(directory + model_file, 'wb') as model_file:
            pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)

        print('Scikit model saved.')

    @staticmethod
    def print_phrase_and_clusters(phrases, cluster_labels, show=1000):
        for i in range(show):
            print(phrases[i], cluster_labels[i])

    def get_clustered_predictions_and_counts(self, predictions, phrases):
        filtered_labels = []

        for i in range(len(predictions)):
            prediction = predictions[i]

            if prediction in self.cluster_labels():
                filtered_labels.append(phrases[i])

        phrases_count = Counter(filtered_labels)

        return phrases_count

    def get_phrases_in_good_clusters(self, predictions, phrases):
        filtered_phrases = []

        for i in range(len(predictions)):
            prediction = predictions[i]

            if prediction in self.get_good_clusters():
                filtered_phrases.append(phrases[i])

        phrases_counts = Counter(filtered_phrases)

        return phrases_counts

    def get_clustered_phrases_in_good_clusters(self, predictions, phrases):
        clustered_predictions = {}
        clustered_predictions_weights = {}

        phrases_count = self.get_clustered_predictions_and_counts(predictions, phrases)

        for i in range(len(predictions)):
            test_prediction = predictions[i]

            if test_prediction in self.get_good_clusters():
                if test_prediction not in clustered_predictions:
                    clustered_predictions[test_prediction] = set()
                    clustered_predictions_weights[test_prediction] = 0

                clustered_predictions[test_prediction].add(phrases[i])
                clustered_predictions_weights[test_prediction] += phrases_count[phrases[i]]

        return clustered_predictions, clustered_predictions_weights

    def print_clustered_predictions(self, predictions, phrases):
        clustered_predictions, clustered_predictions_weights = self.get_clustered_phrases_in_good_clusters(predictions, phrases)

        important_clusters = []
        for key in clustered_predictions_weights.keys():
            important_clusters.append((clustered_predictions_weights[key], clustered_predictions[key]))

        important_clusters.sort(key=lambda tup: tup[0], reverse=True)

        for count, phrase_set in important_clusters:
            print('Printing the cluster with {} mentions'.format(count))

            for key in list(phrase_set):
                print(key)

            print('')
