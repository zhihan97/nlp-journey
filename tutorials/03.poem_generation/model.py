# coding=utf-8
# created by msgi on 2020/3/31 12:28 上午

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess
from dataset import PoetryDataSet
from tokenizer import Tokenizer
from utils import load_config, save_config


class PoemModel:
    def __init__(self, save_path,
                 config_path,
                 max_length=64,
                 min_word_frequency=8,
                 batch_size=64,
                 remove_words=None,
                 init_tokens=None,
                 data_path=None):

        if init_tokens is None:
            init_tokens = ["[PAD]", "[NONE]", "[START]", "[END]"]
        if remove_words is None:
            remove_words = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']
        self.save_path = save_path
        self.config_path = config_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.min_word_frequency = min_word_frequency
        self.remove_words = remove_words
        self.init_tokens = init_tokens
        self.data_path = data_path

        self.model = self.load_model()
        if not self.model:
            if self.data_path:
                self.poetry, self.tokens, self.idx_word, self.word_idx = preprocess(self.data_path, self.max_length,
                                                                                    self.min_word_frequency,
                                                                                    self.remove_words,
                                                                                    self.init_tokens)
                # 初始化tokenizer
                self.tokenizer = Tokenizer(self.tokens)
                # 构建数据集
                self.dataset = PoetryDataSet(self.poetry, self.tokenizer, self.batch_size)
                self.model = self.train_model()
            else:
                print('模型不存在，且数据集为空 ，无法重新训练')

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
        # 进行模型编译（选择优化器、损失函数）
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.sparse_categorical_crossentropy
        )
        # 模型总览
        model.summary()
        return model

    def train_model(self):
        model = self.build_model()
        save_config([self.tokens, self.idx_word, self.word_idx], self.config_path)
        model.fit(
            self.dataset.generator(),
            steps_per_epoch=self.dataset.steps,
            epochs=1
        )

        model.save(self.save_path)
        return model

    def load_model(self):
        try:
            model = load_model(self.save_path)
        except IOError:
            model = None
        return model

    def predict(self, token_ids):
        """
        在概率值为前100的词中选取一个词(按概率分布的方式)
        :return: 一个词的编号(不包含[PAD][NONE][START])
        """
        # 预测各个词的概率分布
        # -1 表示只要对最新的词的预测
        # 3: 表示不要前面几个标记符
        _probas = self.model.predict([token_ids, ])[0, -1, 3:]
        # 按概率降序，取前100
        p_args = _probas.argsort()[-100:][::-1]  # 此时拿到的是索引
        p = _probas[p_args]  # 根据索引找到具体的概率值
        p = p / sum(p)  # 归一
        # 按概率抽取一个
        target_index = np.random.choice(len(p), p=p)
        # 前面预测时删除了前几个标记符，因此编号要补上3位，才是实际在tokenizer词典中的编号
        return p_args[target_index] + 3

    def generate_random_poem(self, tokenizer, model, text=""):
        """
        随机生成一首诗
        :param tokenizer: 分词器
        :param model: 古诗模型
        :param text: 古诗的起始字符串，默认为空
        :return: 一首古诗的字符串
        """
        # 将初始字符串转成token_ids，并去掉结束标记[END]
        token_ids = tokenizer.encode(text)[:-1]
        while len(token_ids) < self.max_length:
            # 预测词的编号
            target = self.predict(token_ids)
            # 保存结果
            token_ids.append(target)
            # 到达END
            if target == tokenizer.end_id:
                break

        return "".join(tokenizer.decode(token_ids))

    def generate_acrostic_poem(self, tokenizer, model, heads):
        """
        生成一首藏头诗
        :param tokenizer: 分词器
        :param model: 古诗模型
        :param heads: 藏头诗的头
        :return: 一首古诗的字符串
        """
        # token_ids，只包含[START]编号
        token_ids = [tokenizer.start_id, ]
        # 逗号和句号标记编号
        punctuation_ids = {tokenizer.token_to_id("，"), tokenizer.token_to_id("。")}
        content = []
        # 为每一个head生成一句诗
        for head in heads:
            content.append(head)
            # head转为编号id，放入列表，用于预测
            token_ids.append(tokenizer.token_to_id(head))
            # 开始生成一句诗
            target = -1
            while target not in punctuation_ids:  # 遇到逗号、句号，说明本句结束，开始下一句
                # 预测词的编号
                target = self.predict(token_ids)
                # 因为可能预测到END，所以加个判断
                if target > 3:
                    # 保存结果到token_ids中，下一次预测还要用
                    token_ids.append(target)
                    content.append(tokenizer.id_to_token(target))

        return "".join(content)
