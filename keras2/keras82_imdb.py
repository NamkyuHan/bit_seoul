from tensorflow.keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

# 소스를 완성하시오 embedding



(x_train, y_train), (x_test, y_test) = reuters.load_data(
        num_words=10000, test_split=0.2
)

print(x_train.shape, x_test.shape) #(8982,) (2246,)
print(y_train.shape, y_test.shape) #(8982,) (2246,)

print('훈련용 뉴스 기사 : {}'.format(len(x_train))) #8982
print('테스트용 뉴스 기사 : {}'.format(len(x_test))) #2246

print(x_train[0])
'''
[1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 111, 16, 369, 186, 90, 67, 7, 
89, 5, 19, 102, 6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 
209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 11, 15, 7, 48, 9, 4579, 1005, 504, 6, 258, 
6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 1245, 90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12]
'''
print(y_train[0]) #3

# print(len(x_train[0])) #87
# print(len(y_train[11])) #59

#y의 카테고리 개수 출력
category = np.max(y_train) +1
print("카테고리 : ", category) #46

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)
'''
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
 '''
# 실습 :
# embeding 모델로 구현

max_len = 100
x_train = pad_sequences(x_train, maxlen=max_len) # 훈련용 뉴스 기사 패딩
x_test = pad_sequences(x_test, maxlen=max_len) # 테스트용 뉴스 기사 패딩

y_train = to_categorical(y_train) # 훈련용 뉴스 기사 레이블의 원-핫 인코딩
y_test = to_categorical(y_test) # 테스트용 뉴스 기사 레이블의 원-핫 인코딩

model = Sequential()
model.add(Embedding(1000, 120))
model.add(LSTM(120))
model.add(Dense(46, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test))




