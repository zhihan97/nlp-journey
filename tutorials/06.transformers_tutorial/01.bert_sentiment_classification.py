# coding=utf-8
# created by msgi on 2020/4/9 2:58 下午

import numpy as np
import pandas as pd
import torch
import transformers as ppb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# 加载数据
df = pd.read_csv('./data/train.tsv', delimiter='\t', header=None)

# 加载训练好的分词器和bert模型
tokenizer = ppb.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = ppb.DistilBertModel.from_pretrained('distilbert-base-uncased')

# 用训练好的分词器将文本数据转换为数字格式
tokenized = df[0].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

# 取最大长度的句子作为训练文本长度，低于这个长度的补0
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

input_ids = torch.tensor(np.array(padded))

# 将数字转为向量形式
with torch.no_grad():
    last_hidden_states = model(input_ids)

# 只取经过编码后的最初[CLS]那一维输出的向量用来分类
features = last_hidden_states[0][:, 0, :].numpy()

labels = df[1]
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

# 直接简单采用逻辑回归进行分类
lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

score = lr_clf.score(test_features, test_labels)

print(score)
