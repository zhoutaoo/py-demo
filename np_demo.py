import numpy as np

a = np.matrix([[1, 2, 3], [4, 5, 6]])
b = np.matrix([[4, 5, 6], [1, 2, 3]])
print(a + b)
print(a - b)

x_data = np.float32(np.random.rand(2, 10)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300
print(x_data)
print(y_data)


