import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 /(1 + np.exp(-x)) # 시그모이드는 1과 0에 수렴 이게 그 수식

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

print("x : ", x)
print("y : ", y)

plt.plot(x, y)
plt.grid()
plt.show()