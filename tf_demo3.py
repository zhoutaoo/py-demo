import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('W'):
            W = tf.Variable(tf.random_normal([in_size, out_size]))
            tf.summary.histogram('W', W)
        with tf.name_scope('b'):
            b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram('b', b)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, W), b)
            tf.summary.histogram('Wx_plus_b', Wx_plus_b)

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        tf.summary.histogram('outputs', outputs)
        return outputs


# 1.准备训练数据
# x_data生成-1到1的1行300列个数组
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# noise生成300行1列的数组
noise = np.random.normal(0, 0.05, x_data.shape)
# y_data为x_data平方-0.5+noise
y_data = np.square(x_data) - 0.5 + noise

# 2.定义输入神经网络层占位参数
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 3.添加隐藏神经网络层
layer1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# 添加输出神经网络层
prediction = add_layer(layer1, 10, 1, n_layer=2, activation_function=None)

# 偏差
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
# 优化器,减少偏差
with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 图形化数据
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.show()

# 4.开始训练
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('view/', sess.graph)
    # 初使化参数
    sess.run(tf.global_variables_initializer())
    for step in range(301):
        sess.run(train, feed_dict={xs: x_data, ys: y_data})
        if step % 50 == 0:
            result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(result, step)
            print(step, sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
