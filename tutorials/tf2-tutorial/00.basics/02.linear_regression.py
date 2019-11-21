# coding=utf-8
# created by msg on 2019/11/21 5:52 下午
import tensorflow as tf

X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])

W = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)

vars = [W, b]
num_epoch = 100

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)
for i in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = X * W + b
        L = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
        print('{}:{}'.format(i, L))
    w_grad, b_grad = tape.gradient(L, vars)
    print(w_grad.numpy(), b_grad.numpy())
    optimizer.apply_gradients(grads_and_vars=zip([w_grad, b_grad], vars))

print(W.numpy(), b.numpy())

