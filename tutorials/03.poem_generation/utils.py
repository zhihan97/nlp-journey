# coding=utf-8
# created by msgi on 2020/3/31 8:23 上午
import pickle


# config写入文件保存
def save_config(config, config_file):
    with open(config_file, 'wb') as f:
        pickle.dump(config, f)


# 加载配置文件
def load_config(config_file):
    try:
        with open(config_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
