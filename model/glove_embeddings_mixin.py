from model.base_embeddings_mixin import BaseEmbeddingsMixin

__author__ = 'georgi.val.stoyan0v@gmail.com'

class GloveMixin(BaseEmbeddingsMixin):
    _MODEL_NAME = "GloVe"

    def __init__(self, directory, file_name):
        super().__init__(directory, file_name)
        self._model_name = self._MODEL_NAME
