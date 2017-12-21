import pandas as pd
from sklearn import cluster
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from model.base_model import BaseModel
from model.glove_embeddings_mixin import GloveMixin

__author__ = 'georgi.val.stoyan0v@gmail.com'


class AffinityClusterModel(BaseModel, GloveMixin):
    _PHRASE_KEY = 'Phrase'
    _LABEL_KEY = 'Label'
    _LIST_ID_KEY = 'List Id'
    _DIRECTORY = '/mnt/c/Users/gvs/ubuntu/neural-review-summarization/model/'
    _AFFINITY_CLUSTER_CENTERS = 'cluster_centers.csv'
    _AFFINITY_CLUSTER_LABELS = 'phrase_cluster_labels.csv'

    _GOOD_CLUSTERS = {570, 455, 35, 190, 404, 26, 182, 281, 226, 84, 476, 207, 487, 131, 520, 469, 28, 59, 312, 329, 351,
                      1, 225, 269, 320, 359, 409, 24, 26, 76, 88, 210, 386, 464, 467}
    _BAD_CLUSTERS = {310, 337, 260, 365, 368, 435, 510, 522, 239, 513, 77, 435, 130, 195, 306, 29, 516, 383, 384, 68, 71,
                     277, 433}

    def __init__(self, emb_dir=None, emb_file=None):
        super().__init__(emb_dir, emb_file)

        self.phrase_cluster_labels = None
        self.cluster_centers = None

        self.all_cluster_centers = None
        self.all_cluster_center_meanings = None

        self._nearest_n_model = NearestNeighbors(metric='cosine')

    def _init_model(self):
        self.load_embeddings_model()

        return cluster.AffinityPropagation(damping=0.9, max_iter=2000, convergence_iter=1000, preference=None,
                                           affinity='precomputed', verbose=True)

    def train_model(self, n_grams):
        print('Training started..')
        self._generate_n_gram_meanings(n_grams)
        self._generate_meaning_matrices()

        self._model.fit_predict(self.n_gram_meaning_matrix)
        print('Training completed!')

        all_cluster_center_meanings = self.save_model()

        self._nearest_n_model.fit(pd.np.array(all_cluster_center_meanings))

    def predict(self, x):
        vectorized_phrases, labels = self.vectorize_phrases(x)
        predictions = self._nearest_n_model.kneighbors(vectorized_phrases, return_distance=False)[:, 0]

        return predictions, labels

    def save_model(self, save_dir=_DIRECTORY, cluster_labels_file=_AFFINITY_CLUSTER_LABELS,
                   cluster_centers_file=_AFFINITY_CLUSTER_CENTERS):
        print('Saving model..')
        phrase_cluster_labels_df = pd.DataFrame(list(zip(self.n_gram_meaning_labels, self.cluster_labels())),
                                                columns=[self._PHRASE_KEY, self._LABEL_KEY])
        cluster_centers = [(center, self.n_gram_meaning_labels[center], self.cluster_labels()[center]) for center in
                           self._model.cluster_centers_indices_]
        cluster_centers_df = pd.DataFrame(cluster_centers, columns=[self._LIST_ID_KEY, self._PHRASE_KEY, self._LABEL_KEY])

        phrase_cluster_labels_df.to_csv(save_dir + cluster_labels_file, encoding='utf-8', index=False)
        cluster_centers_df.to_csv(save_dir + cluster_centers_file, encoding='utf-8', index=False)

        self.all_cluster_centers = list(cluster_centers_df[self._PHRASE_KEY])
        self.all_cluster_center_meanings = [self.vectorize_phrase(cluster_center_phrase) for
                                            cluster_center_phrase in tqdm(self.all_cluster_centers)]
        print('Model saved.')
        return self.all_cluster_center_meanings

    def load_model(self, load_dir=_DIRECTORY, cluster_centers_file=_AFFINITY_CLUSTER_CENTERS):
        cluster_centers_df = pd.read_csv(load_dir + cluster_centers_file)

        self.all_cluster_centers = list(cluster_centers_df[self._PHRASE_KEY])
        self.all_cluster_center_meanings = [self.vectorize_phrase(cluster_center_phrase) for
                                            cluster_center_phrase in tqdm(self.all_cluster_centers)]

        print('Model loaded.')
        return self.all_cluster_center_meanings

    def good_clusters(self):
        return self._GOOD_CLUSTERS