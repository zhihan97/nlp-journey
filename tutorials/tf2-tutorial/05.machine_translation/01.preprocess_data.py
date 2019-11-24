# coding=utf-8
# created by msg on 2019/11/24 2:30 下午

import re
import unicodedata


# 第一个参数指定字符串标准化的方式。
# NFC表示字符应该是整体组成(比如可能的话就使用单一编码)
# NFD表示字符应该分解为多个组合字符表示。
# unicodedata.category(chr) 把一个字符返回它在UNICODE里分类的类型
# unicode file 2 ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


# 预处理
def preprocess_sentence(w):
    # 转小写换编码
    w = unicode_to_ascii(w.lower().strip())

    # 标点符号前后加上空格
    #  eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)

    # 去掉多余空格
    w = re.sub(r'[" ]+', " ", w)

    # 去掉前后空格
    w = w.rstrip().strip()

    # 加开始和结束字符
    w = '<start> ' + w + ' <end>'
    return w


# 创建数据集
def parse_data(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    sentence_pairs = [line.split('\t') for line in lines]
    preprocessed_sentence_pairs = [(preprocess_sentence(en), preprocess_sentence(spa)) for en, spa in sentence_pairs]
    return zip(*preprocessed_sentence_pairs)


if __name__ == '__main__':
    spa_eng_path = './spa-eng/spa.txt'
    en = 'Come on.'
    spa = 'Ándale.'

    print(unicode_to_ascii(en))
    print(unicode_to_ascii(spa))

    print(preprocess_sentence(en))
    print(preprocess_sentence(spa))

    en_dataset, spa_dataset = parse_data(spa_eng_path)

    print(en_dataset[-1])
    print(spa_dataset[-1])
