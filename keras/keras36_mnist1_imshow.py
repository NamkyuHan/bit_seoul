import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print(x_train[0])
print(y_train[0])

# 결과값
# (60000, 28, 28) (10000, 28, 28)
# (60000,) (10000,)

# 10개의 그림중 내가 마음에 드는 그림을 찾아라
# 0~9가 평등하다는 조건을 갖춰야 한다
plt.imshow(x_train[0], 'gray')
plt.show()


