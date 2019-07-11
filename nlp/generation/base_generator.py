from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop
import numpy as np
import random
import sys


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


class TextGenerator:

    def __init__(self):
        pass

    def train(self):
        model = self.build_model()
        x, y, chars, text,char_indices = self.preprocess()
        for epoch in range(1, 60):
            print('epoch', epoch)
            model.fit(x, y,
                      batch_size=128,
                      epochs=1)
            start_index = random.randint(0, len(text) - 60 - 1)
            generated_text = text[start_index: start_index + 60]
            print('随机文本:' + generated_text)

            for temperature in [0.2, 0.5, 1.0, 1.2]:
                print('temperature:', temperature)
                sys.stdout.write(generated_text)

                # 生成100个字
                for i in range(100):
                    sampled = np.zeros((1, 60, len(chars)))
                    for t, char in enumerate(generated_text):
                        sampled[0, t, char_indices[char]] = 1.

                    preds = model.predict(sampled, verbose=0)[0]
                    next_index = sample(preds, temperature)
                    next_char = chars[int(next_index)]

                    generated_text += next_char
                    generated_text = generated_text[1:]

                    sys.stdout.write(next_char)
                    sys.stdout.flush()
                print()

    def build_model(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(60, 2539)))
        model.add(Dense(2539, activation='softmax'))
        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    def generate(self):
        pass

    def save_model(self):
        pass

    def load_model(self):

        pass

    def preprocess(self):
        with open('/home/msg/workspace/pythons/nlp-journey/demos/data/tianlong.txt', encoding='utf-8') as txt:
            text = txt.read()[:100000]
        maxlen = 60
        step = 10
        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])
        chars = sorted(list(set(text)))
        char_indices = dict((char, chars.index(char)) for char in chars)
        x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1
        return x, y, chars,text,char_indices


if __name__ == '__main__':
    generator = TextGenerator()
    generator.train()
