#1. 데이터
import numpy as np
x = np.array([range(1, 101), range(711,811), range(100)])
y = np.array([range(101, 201), range(311,411), range(100)])

# [] 이걸로 바꾸면 일단 3,100으로 나온다


# x = np.transpose(x)
# x = x.transpose()

# y = np.transpose(y)

x = np.transpose(x)

# print(x[1][10])
print(x[10][1])
print(x.shape) #(3, )
# print(x.reshape(-1,1))
# (100,3)

