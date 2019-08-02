# tensorflow学习记录

## truncated_normal

tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)        

从截断的正态分布中输出随机值。 shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，均值和标准差自己设定。随机产生的值如果与均值的差值大于两倍的标准差，那就重新生成。