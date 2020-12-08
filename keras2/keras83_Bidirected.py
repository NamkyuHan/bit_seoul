from tensorflow.keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 소스를 완성하시오 embedding



(x_train, y_train), (x_test, y_test) = reuters.load_data(
        num_words=10000, test_split=0.2
)

print(x_train.shape, x_test.shape) #(8982,) (2246,)
print(y_train.shape, y_test.shape) #(8982,) (2246,)

print(x_train[0])
print(y_train[0])

print(len(x_train[0])) #87
print(len(y_train[11])) #59

#y의 카테고리 개수 출력
category = np.max(y_train) +1
print("카테고리 : ", category) #46

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

# 실습 :
# embeding 모델로 구현






