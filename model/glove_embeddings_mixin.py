from model.base_embeddings_mixin import BaseEmbeddingsMixin

__author__ = 'georgi.val.stoyan0v@gmail.com'

class GloveMixin(BaseEmbeddingsMixin):
    _EMB_MODEL_NAME = 'GloVe'
    _EMB_DIRECTORY = '/home/gvs/neural-review-summarization/model/embeddings/'
    _EMB_FILE_NAME = 'glove.6B.300d.txt'

    def __init__(self, directory=_EMB_DIRECTORY, file_name=_EMB_FILE_NAME):
        super().__init__(directory=directory, file_name=file_name)
        self._emb_model_name = self._EMB_MODEL_NAME
