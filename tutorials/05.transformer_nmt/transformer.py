# coding=utf-8
# created by msgi on 2020/4/2 11:25 上午

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt


# examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
# train_examples, val_examples = examples['train'], examples['validation']
#
# tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#     (en.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)
#
# tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#     (pt.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)
#
# sample_string = 'Transformer is awesome.'
#
# tokenized_string = tokenizer_en.encode(sample_string)
# print('Tokenized string is {}'.format(tokenized_string))
#
# original_string = tokenizer_en.decode(tokenized_string)
# print('The original string is {}'.format(original_string))


def get_angle(position, i, d_model):
    rate = 1 / np.power(10000, 2 * (i // 2) / np.float32(d_model))
    # (length, d_model)
    return position * rate


def positional_encoding(position, d_model):
    position = np.arange(position)[:, np.newaxis]
    d_model_batch = np.arange(d_model)[np.newaxis, :]
    print('position shape: ', position.shape)
    print('d_model shape: ', d_model_batch.shape)
    angle_rads = get_angle(position,  # (length, 1)
                           d_model_batch,  # (1, d_model)
                           d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


pos_encoding = positional_encoding(50, 512)
print(pos_encoding.shape)

plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()
