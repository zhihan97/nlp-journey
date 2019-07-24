import os
import tensorflow as tf

# 关闭TensorFlow的warning异常信息：可以把下边代码注释掉，看看效果
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义计算图
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

addition = tf.add(X, Y, name="addition")

# 创建session
with tf.Session() as session:
    result = session.run(addition, feed_dict={X: [1, 2, 10], Y: [4, 2, 10]})
    print(result)
