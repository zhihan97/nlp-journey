# coding=utf-8
# created by msgi on 2020/3/31 12:28 上午

import tensorflow as tf
from tensorflow.keras.models import load_model


class PoemModel:
    def __init__(self, tokenizer, dataset, save_path):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.save_path = save_path

        self.model = self.load_model()
        if not self.model:
            self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            # 词嵌入层
            tf.keras.layers.Embedding(input_dim=self.tokenizer.dict_size, output_dim=150),
            # 第一个LSTM层
            tf.keras.layers.LSTM(150, dropout=0.5, return_sequences=True),
            # 第二个LSTM层
            tf.keras.layers.LSTM(150, dropout=0.5, return_sequences=True),
            # 利用TimeDistributed对每个时间步的输出都做Dense操作(softmax激活)
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.tokenizer.dict_size, activation='softmax')),
        ])
        # 模型总览
        model.summary()
        # 进行模型编译（选择优化器、损失函数）
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.sparse_categorical_crossentropy
        )
        model.fit(
            self.dataset.generator(),
            steps_per_epoch=self.dataset.steps,
            epochs=10
        )
        model.save(self.save_path)
        return model

    def load_model(self):
        try:
            model = load_model(self.save_path)
        except IOError:
            model = None
        return model
