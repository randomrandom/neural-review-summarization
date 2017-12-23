import random

import pandas as pd
from sklearn import cluster
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from model.base_model import BaseModel
from model.glove_embeddings_mixin import GloveMixin
from processors.base_processor import BaseProcessor

__author__ = 'georgi.val.stoyan0v@gmail.com'


class AffinityClusterModel(BaseModel, GloveMixin):
    _PHRASE_KEY = 'Phrase'
    _CLUSTER_LABEL_KEY = 'Cluster Label'
    _LIST_ID_KEY = 'List Id'
    _NA_KEY = 'NA'
    _GOOD_WORDS_KEY = 'Good Words'
    _SCORE_KEY = 'Score'
    _GOOD_WORD_KEY = 'GW{}'
    _WORD_KEY = 'W{}'
    _GOOD_CLUSTER_THRESHOLD = 0.2
    _DIRECTORY = '/mnt/c/Users/gvs/ubuntu/neural-review-summarization/model/saved/'
    _AFFINITY_CLUSTER_CENTERS = 'cluster_centers.csv'
    _AFFINITY_CLUSTER_LABELS = 'phrase_cluster_labels.csv'
    _AFFINITY_MODEL_PKL = '_affinity_model.pkl'
    _NEAREST_N_MODEL_PKL = '_nearest_n_model.pkl'

    def __init__(self):
        super().__init__()

        self.phrase_cluster_labels = None
        self.cluster_centers = None

        self.all_cluster_centers = None
        self.all_cluster_center_meanings = None
        self._good_clusters = None

        self._nearest_n_model = NearestNeighbors(metric='cosine')

    def _init_model(self):
        self.load_embeddings_model()

        return cluster.AffinityPropagation(damping=0.9, max_iter=2000, convergence_iter=1000, preference=None,
                                           affinity='precomputed', verbose=True)

    def train_model(self, n_grams, model_name):
        print('Training started..')
        self._generate_n_gram_meanings(n_grams)
        self._generate_meaning_matrices()

        self._model.fit_predict(self.n_gram_meaning_matrix)
        print('Training completed!')

        all_cluster_center_meanings = self.save_model(model_name)

        self._nearest_n_model.fit(pd.np.array(all_cluster_center_meanings))

    def predict(self, x):
        vectorized_phrases, original_phrases = self.vectorize_phrases(x)
        cluster_center_predictions = self._nearest_n_model.kneighbors(vectorized_phrases, return_distance=False)[:, 0]

        predictions = [self.phrase_cluster_labels[self.all_cluster_centers[cluster_center_prediction]]
                       for cluster_center_prediction in cluster_center_predictions]

        closest_phrases = [self.all_cluster_centers[cluster_center_prediction]
                           for cluster_center_prediction in cluster_center_predictions]

        return predictions, original_phrases, closest_phrases

    def __add_cluster_samples(self, phrase_cluster_labels_df, cluster_centers_df, sample_size=5):
        for i in range(len(cluster_centers_df)):
            cluster_samples = list(
                phrase_cluster_labels_df[phrase_cluster_labels_df[self._CLUSTER_LABEL_KEY] == i][self._PHRASE_KEY]
            )
            cluster_samples.extend([self._NA_KEY] * (sample_size - len(cluster_samples)))
            sample = random.sample(cluster_samples, sample_size)

            for j in range(len(sample)):
                cluster_centers_df.loc[i, self._WORD_KEY.format(j)] = sample[j]

    def save_model(self, model_name, save_dir=_DIRECTORY, cluster_labels_file=_AFFINITY_CLUSTER_LABELS,
                   cluster_centers_file=_AFFINITY_CLUSTER_CENTERS):
        print('Saving model..')
        phrase_cluster_labels_df = pd.DataFrame(list(zip(self.n_gram_meaning_labels, self.cluster_labels())),
                                                columns=[self._PHRASE_KEY, self._CLUSTER_LABEL_KEY])
        cluster_centers = [(center, self.n_gram_meaning_labels[center], self.cluster_labels()[center]) for center in
                           self._model.cluster_centers_indices_]
        cluster_centers_df = pd.DataFrame(cluster_centers,
                                          columns=[self._LIST_ID_KEY, self._PHRASE_KEY, self._CLUSTER_LABEL_KEY])

        # generate cluster samples
        sample_size = 5
        self.__add_cluster_samples(phrase_cluster_labels_df, cluster_centers_df, sample_size=sample_size)
        cluster_centers_df[self._GOOD_WORDS_KEY] = 0
        cluster_centers_df[self._SCORE_KEY] = 0
        for i in range(sample_size):
            cluster_centers_df[self._GOOD_WORD_KEY.format(i)] = 0

        phrase_cluster_labels_df.to_csv(save_dir + model_name + '_' + cluster_labels_file, encoding='utf-8',
                                        index=False)
        cluster_centers_df.to_csv(save_dir + model_name + '_' + cluster_centers_file, encoding='utf-8', index=False)

        self.all_cluster_centers = list(cluster_centers_df[self._PHRASE_KEY])
        self.all_cluster_center_meanings = [self.vectorize_phrase(cluster_center_phrase) for
                                            cluster_center_phrase in tqdm(self.all_cluster_centers)]

        self.phrase_cluster_labels = BaseProcessor.df_to_dict(phrase_cluster_labels_df, value_column=self._CLUSTER_LABEL_KEY)

        self._save_scikit_model(self._model, save_dir, model_name + self._AFFINITY_MODEL_PKL)

        print('Model saved.')
        return self.all_cluster_center_meanings

    def __find_good_clusters(self, cluster_centers_df):
        good_clusters = {}

        for i in range(len(cluster_centers_df)):
            cluster_label = cluster_centers_df.iloc[i][self._CLUSTER_LABEL_KEY]
            cluster_score = cluster_centers_df.iloc[i][self._SCORE_KEY]

            if cluster_score >= self._GOOD_CLUSTER_THRESHOLD:
                good_clusters[cluster_label] = cluster_score

        return good_clusters

    def load_model(self, model_name, load_dir=_DIRECTORY, cluster_labels_file=_AFFINITY_CLUSTER_LABELS,
                   cluster_centers_file=_AFFINITY_CLUSTER_CENTERS):
        phrase_cluster_labels_df = pd.read_csv(load_dir + model_name + '_' + cluster_labels_file)
        cluster_centers_df = pd.read_csv(load_dir + model_name + '_' + cluster_centers_file)

        self._good_clusters = self.__find_good_clusters(cluster_centers_df)
        self.all_cluster_centers = list(cluster_centers_df[self._PHRASE_KEY])
        self.all_cluster_center_meanings = [self.vectorize_phrase(cluster_center_phrase) for
                                            cluster_center_phrase in tqdm(self.all_cluster_centers)]

        self.phrase_cluster_labels = BaseProcessor.df_to_dict(phrase_cluster_labels_df, value_column=self._CLUSTER_LABEL_KEY)

        self._model = self._load_scikit_model(load_dir, model_name + self._AFFINITY_MODEL_PKL)

        self._nearest_n_model = NearestNeighbors(metric='cosine')
        self._nearest_n_model.fit(pd.np.array(self.all_cluster_center_meanings))

        print('Model loaded.')

    def get_good_clusters(self):
        return self._good_clusters
