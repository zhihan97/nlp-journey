# coding=utf-8
# created by msgi on 2020/1/21 2:06 下午

import numpy as np
from functools import reduce


def sigmoid(input_x):
    """
    sigmoid 函数实现
    :param input_x: 输入向量
    :return: 对输入向量作用 sigmoid 函数之后得到的输出
    """
    return 1.0 / (1 + np.exp(-input_x))


class Node(object):
    """
    神经网络的节点类
    """
    def __init__(self, layer_index, node_index):
        """
        初始化一个节点
        :param layer_index: 层的索引，也就是表示第几层
        :param node_index: 节点的索引，也就是表示节点的索引
        """
        self.layer_index = layer_index
        self.node_index = node_index
        # 设置此节点的下游节点，也就是这个节点与下一层的哪个节点相连
        self.downstream = []
        # 设置此节点的上游节点，也就是哪几个节点的下游节点与此节点相连
        self.upstream = []
        # 此节点的输出
        self.output = 0
        # 此节点真实值与计算值之间的差值
        self.delta = 0

    def set_output(self, output):
        """
        设置节点的 output
        :param output: 节点的 output
        :return:
        """
        self.output = output

    def append_downstream_connection(self, conn):
        """
        添加此节点的下游节点的连接
        :param conn: 当前节点的下游节点的连接的 list
        :return:
        """
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        """
        添加此节点的上游节点的连接
        :param conn: 当前节点的上游节点的连接的 list
        :return:
        """
        self.upstream.append(conn)

    def calc_output(self):
        """
        计算节点的输出，依据 output = sigmoid(wTx)
        :return:
        """
        # 使用 reduce() 函数对其中的因素求和
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        # 对上游节点的 output 乘 weights 之后求和得到的结果应用 sigmoid 函数，得到当前节点的 output
        self.output = sigmoid(output)

