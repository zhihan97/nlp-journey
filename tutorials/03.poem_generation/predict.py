# coding=utf-8
# created by msgi on 2020/3/31 8:17 上午
import numpy as np
from tokenizer import Tokenizer
from model import PoemModel
from utils import load_config


def predict(model, token_ids):
    """
    在概率值为前100的词中选取一个词(按概率分布的方式)
    :return: 一个词的编号(不包含[PAD][NONE][START])
    """
    # 预测各个词的概率分布
    # -1 表示只要对最新的词的预测
    # 3: 表示不要前面几个标记符
    _probas = model.predict([token_ids, ])[0, -1, 3:]
    # 按概率降序，取前100
    p_args = _probas.argsort()[-100:][::-1]  # 此时拿到的是索引
    p = _probas[p_args]  # 根据索引找到具体的概率值
    p = p / sum(p)  # 归一
    # 按概率抽取一个
    target_index = np.random.choice(len(p), p=p)
    # 前面预测时删除了前几个标记符，因此编号要补上3位，才是实际在tokenizer词典中的编号
    return p_args[target_index] + 3


def generate_random_poem(tokenizer, model, text=""):
    """
    随机生成一首诗
    :param tokenizer: 分词器
    :param model: 古诗模型
    :param text: 古诗的起始字符串，默认为空
    :return: 一首古诗的字符串
    """
    # 将初始字符串转成token_ids，并去掉结束标记[END]
    token_ids = tokenizer.encode(text)[:-1]
    while len(token_ids) < MAX_LEN:
        # 预测词的编号
        target = predict(model, token_ids)
        # 保存结果
        token_ids.append(target)
        # 到达END
        if target == tokenizer.end_id:
            break

    return "".join(tokenizer.decode(token_ids))


tokens, idx_word, word_idx = load_config('config/config.pkl')

# 初始化tokenizer
tokenizer = Tokenizer(tokens)

# 训练（有模型的时候直接家在）
model = PoemModel(tokenizer, 'model')

token_ids = tokenizer.encode("清风明月")[:-1]
while len(token_ids) < 13:
    # 预测词的编号
    target = predict(model, token_ids)
    # 保存结果
    token_ids.append(target)
    # 到达END
    if target == tokenizer.end_id:
        break

print("".join(tokenizer.decode(token_ids)))
