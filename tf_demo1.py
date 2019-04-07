import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# 1.准备数据
x_data = np.random.rand(100).astype(np.float32)
# 实际结果
y_data = 0.1 * x_data + 0.3

# 2.创建tf结构
# W取值范围
W = tf.Variable(tf.random_uniform([1], -1, 1))
# b先从0开始
b = tf.Variable(tf.zeros([1]))
# 模型
y = W * x_data + b

# 方差
loss = tf.reduce_mean(tf.square(y - y_data))
# 优化器
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 使用优化器减少偏差
train = optimizer.minimize(loss)

# 图形化数据
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# 点图
ax.scatter(x_data, y_data)
#plt.ion()
plt.show()

# 3.训练
with tf.Session() as session:
    # 初使化变量
    session.run(tf.global_variables_initializer())
    for step in range(201):
        # 训练
        session.run(train)
        if step % 20 == 0:
            print(step, session.run(W), session.run(b))
            lines = ax.plot(x_data, session.run(y), 'ob')
            plt.pause(0.1)

