# coding=utf-8
# created by msg on 2019/11/23 12:49 下午

import tensorflow as tf


class MeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))
