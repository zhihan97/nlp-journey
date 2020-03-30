# coding=utf-8
# created by msgi on 2020/3/30 3:52 下午

import re
from collections import Counter


def preprocess(data_path, max_length, min_word_frequency, remove_words, init_tokens):
    # 一首诗（一行）对应一个列表的元素
    poetry = []
    # 按行读取数据 poetry.txt
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 遍历处理每一条数据
    for line in lines:
        # 利用正则表达式拆分标题和内容
        fields = re.split(r"[:：]", line)
        # 跳过异常数据
        if len(fields) != 2:
            continue
        # 得到诗词内容（后面不需要标题）
        content = fields[1]
        # 跳过内容过长的诗词
        if len(content) > max_length - 2:
            continue
        # 跳过存在禁用符的诗词
        if any(word in content for word in remove_words):
            continue

        poetry.append(content.replace('\n', ''))  # 最后要记得删除换行符

    # 统计词频，利用Counter可以直接按单个字符进行统计词频
    counter = Counter()
    for line in poetry:
        counter.update(line)
    # 过滤掉低词频的词
    tokens = [token for token, count in counter.items() if count >= min_word_frequency]

    tokens = init_tokens + tokens

    # 映射: 词 -> 编号
    word_idx = {}
    # 映射: 编号 -> 词
    idx_word = {}
    for idx, word in enumerate(tokens):
        word_idx[word] = idx
        idx_word[idx] = word
    return poetry, tokens, idx_word, word_idx
