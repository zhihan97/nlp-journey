# coding=utf-8
# created by msgi on 2020/4/25 10:01 上午

import numpy as np
import logging
from gensim.models import KeyedVectors
from smartnlp.utils.basic_log import Log

log = Log(logging.INFO)


# 加载词向量(获取需要的词向量)
def load_bin_word2vec(word_index, word2vec_path):
    log.info('Begin load word vector from bin...')
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    embeddings = _select_vectors(word2vec, word_index)
    log.info('End load word vector from bin...')
    return embeddings


def load_text_vector(word_index, word2vec_path):
    log.info('Begin load word vector from text...')
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
    embeddings = _select_vectors(word2vec, word_index)
    log.info('End load word vector from text...')
    return embeddings


def _select_vectors(word2vec, word_index):
    embedding_dim = word2vec.vector_size
    embeddings = 1 * np.random.randn(len(word_index) + 1, embedding_dim)
    embeddings[0] = 0

    for word, index in word_index.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)
    return embeddings
