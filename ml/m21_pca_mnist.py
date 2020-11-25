import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
# print(x.shape) #(70000, 28, 28)

# 실습 
# PCA를 통해 마구마구 0.95 이상인게 몇개?
# PCA 배운거 다 집어넣고 확인

x = x.reshape(70000, 28*28).astype('float32')/255. 

pca = PCA(n_components=154) # 내가 출력하고 싶은 열을 설정할 수 있다 차원축소
x2d = pca.fit_transform((x))
print(x2d.shape) #(70000, 9)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR) #1.
print(sum(pca_EVR)) #2.
'''
#2
0.9481826884444571
'''

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

d = np.argmax(cumsum >= 0.95) +1
print(cumsum >= 0.95)
print(d) #우리가 원하는 n_components의 개수

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

