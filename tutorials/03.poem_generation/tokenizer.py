# coding=utf-8
# created by msgi on 2020/3/30 3:51 下午


class Tokenizer:
    """
    分词器
    """

    def __init__(self, tokens):
        # 词汇表大小
        self.dict_size = len(tokens)
        # 生成映射关系
        self.token_id = {}  # 映射: 词 -> 编号
        self.id_token = {}  # 映射: 编号 -> 词
        for idx, word in enumerate(tokens):
            self.token_id[word] = idx
            self.id_token[idx] = word

        # 各个特殊标记的编号id，方便其他地方使用
        self.start_id = self.token_id["[START]"]
        self.end_id = self.token_id["[END]"]
        self.none_id = self.token_id["[NONE]"]
        self.pad_id = self.token_id["[PAD]"]

    def id_to_token(self, token_id):
        """
        编号 -> 词
        """
        return self.id_token.get(token_id)

    def token_to_id(self, token):
        """
        词 -> 编号
        """
        return self.token_id.get(token, self.none_id)

    def encode(self, tokens):
        """
        词列表 -> [START]编号 + 编号列表 + [END]编号
        """
        token_ids = [self.start_id, ]  # 起始标记
        # 遍历，词转编号
        for token in tokens:
            token_ids.append(self.token_to_id(token))
        token_ids.append(self.end_id)  # 结束标记
        return token_ids

    def decode(self, token_ids):
        """
        编号列表 -> 词列表(去掉起始、结束标记)
        """
        # 起始、结束标记
        flag_tokens = {"[START]", "[END]"}

        tokens = []
        for idx in token_ids:
            token = self.id_to_token(idx)
            # 跳过起始、结束标记
            if token not in flag_tokens:
                tokens.append(token)
        return tokens
