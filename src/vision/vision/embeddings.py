import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function


@register_embedding_function
class MultilingualONNXEmbedding(EmbeddingFunction[Documents]):
    """Embeddings multilingües (familia e5) en ONNX, cargados desde disco sin internet.

    all-MiniLM-L6-v2 solo entiende inglés: con preguntas en español todas las
    distancias quedaban casi iguales y no se podía filtrar por umbral.
    """

    def __init__(self, model_path, query_prefix='query: ', doc_prefix='passage: ', max_length=512):
        self.model_path = model_path
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.max_length = max_length

        self.tokenizer = Tokenizer.from_file(os.path.join(model_path, 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding()
        self.session = ort.InferenceSession(
            os.path.join(model_path, 'onnx', 'model.onnx'), providers=['CPUExecutionProvider'])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def _embed(self, textos):
        enc = self.tokenizer.encode_batch(list(textos))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {'input_ids': ids, 'attention_mask': mask}
        if 'token_type_ids' in self.input_names:
            feeds['token_type_ids'] = np.zeros_like(ids)
        hidden = self.session.run(None, feeds)[0]

        # mean pooling sobre los tokens reales y normalización L2
        m = mask[..., None].astype(np.float32)
        vec = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        return [v.astype(np.float32) for v in vec]

    def __call__(self, input: Documents) -> Embeddings:
        return self._embed(self.doc_prefix + t for t in input)

    def embed_query(self, input: Documents) -> Embeddings:
        return self._embed(self.query_prefix + t for t in input)

    @staticmethod
    def name():
        return 'multilingual_onnx'

    def get_config(self):
        return {
            'model_path': self.model_path,
            'query_prefix': self.query_prefix,
            'doc_prefix': self.doc_prefix,
            'max_length': self.max_length,
        }

    @staticmethod
    def build_from_config(config):
        return MultilingualONNXEmbedding(**config)
