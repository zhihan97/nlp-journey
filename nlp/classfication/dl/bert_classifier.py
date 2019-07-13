from keras import Input, Model
from keras.layers import Lambda, Dense
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs


class ZhTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


class BertTextClassifier:
    """
    目前示例只是拿谷歌训练好的模型来使用，不涉及训练过程（训练成本太高）
    """

    def __init__(self, config_path,
                 checkpoint_path,
                 dict_path):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.dict_path = dict_path

    def train(self):
        model = self.build_model()

        pass

    def build_model(self):
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path)

        for l in bert_model.layers:
            l.trainable = True

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))

        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
        p = Dense(1, activation='sigmoid')(x)

        model = Model([x1_in, x2_in], p)
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(1e-5),  # 用足够小的学习率
            metrics=['accuracy'])
        model.summary()
        return model

    def preprocess(self):
        token_dict = {}
        with codecs.open(self.dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        tokenizer = ZhTokenizer(token_dict)
        tokenizer.tokenize(u'今天天气不错')
