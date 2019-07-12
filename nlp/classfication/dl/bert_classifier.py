from keras import Input, Model
from keras.layers import Lambda, Dense
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint


class BertTextClassifier:

    def build_model(self):
        bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

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
