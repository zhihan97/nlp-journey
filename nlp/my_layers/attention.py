# coding=utf-8
# created by msg on 2019/11/21 4:01 下午

import tensorflow as tf
from tensorflow.keras import regularizers, constraints
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer


def dot_product(x, kernel):
    return tf.squeeze(tf.matmul(x, tf.expand_dims(kernel)), axis=-1)


class Attention(Layer):
    def __init__(self, w_regularizer=None,
                 u_regularizer=None,
                 b_regularizer=None,
                 w_constraint=None,
                 u_constraint=None,
                 b_constraint=None,
                 bias=True,
                 **kwargs):
        """
        自定义的用keras写的attention层
        """
        self.supports_masking = True
        self.init = glorot_uniform

        self.w_regularizer = regularizers.get(w_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.w_constraint = constraints.get(w_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.w = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.w_regularizer,
                                 constraint=self.w_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.w)
        if self.bias:
            uit += self.b
        uit = tf.tanh(uit)

        ait = dot_product(uit, self.u)
        a = tf.exp(ait)

        if mask is not None:
            a *= tf.cast(mask, tf.float32)
        a /= tf.cast(tf.reduce_sum(a, axis=1, keepdims=True) + (), tf.float32)

        a = tf.expand_dims(a)
        weighted_input = x * a
        return tf.reduce_sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
