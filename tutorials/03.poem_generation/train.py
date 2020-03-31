# coding=utf-8
# created by msgi on 2020/3/31 12:31 上午

from model import PoemModel

# 数据路径
DATA_PATH = './data/poetry.txt'
CONFIG_PATH = './config/config.pkl'

# 训练（有模型的时候直接家在）
model = PoemModel(DATA_PATH, CONFIG_PATH)
