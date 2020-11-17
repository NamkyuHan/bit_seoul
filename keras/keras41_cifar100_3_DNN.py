#이미지 종류가 10개
#1. 데이터

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# x_predict = x_test[:10] 내가 답을 유추해 볼건데 10개 까지 볼거야(슬라이싱)
# y_real = y_test[:10] 내가 답을 유추해 볼거면 답이 있어야 겠지? 정해져 있는 그 답도 10개 까지 볼거야(슬라이싱)
x_predict = x_test[:10]
y_real = y_test[:10]

# 프린트 해서 값을 쉐이프를 확인해보자
# x_train, x_test, y_train, y_test
# 쉐이프 확인 했으면 헷갈리지 않게 옆에다가 잘 써주자
# print(x_train[0]), print(y_train[0]) = 이건 왜? 원래 매칭되어 있는 값을 내눈으로 한번 더 볼려고
print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)
print("x_train[0] : ",x_train[0])
'''
x_train[0] :  [[[255 255 255]
  [255 255 255]
  [255 255 255]
  ...
  [195 205 193]
  [212 224 204]
  [182 194 167]]

 [[255 255 255]
  [254 254 254]
  [254 254 254]
  ...
  [170 176 150]
  [161 168 130]
  [146 154 113]]

 [[255 255 255]
  [254 254 254]
  [255 255 255]
  ...
  [189 199 169]
  [166 178 130]
  [121 133  87]]

 ...

 [[148 185  79]
  [142 182  57]
  [140 179  60]
  ...
  [ 30  17   1]
  [ 65  62  15]
  [ 76  77  20]]

 [[122 157  66]
  [120 155  58]
  [126 160  71]
  ...
  [ 22  16   3]
  [ 97 112  56]
  [141 161  87]]

 [[ 87 122  41]
  [ 88 122  39]
  [101 134  56]
  ...
  [ 34  36  10]
  [105 133  59]
  [138 173  79]]]
y_train[0] :  [19]
'''
# plt.imshow(x_train[0], 'gray')
# plt.show()

# 2. OneHotEncodeing  인코딩은 아래 코드와 같이 케라스에서 제공하는 “to_categorical()”로 쉽게 처리할 수 있습니다
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# OneHotEncodeing으로 변경된 y_train, y_test 쉐이프 확인하기
# 대입되어 있는 y_train 값 확인하기
print("y_train shape : ", y_train.shape) #(50000, 10)
print("y_test shape : ", y_test.shape) #(10000, 10)
print("y_train data : ", y_train[0])
'''
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0.] 
'''
# DNN에 집어넣기 위해 2차원으로 x_train, x_test, x_predict reshape하기
# x_predict은 x_test를 통해 위에 10개까지 본다고 설정해 놨기 때문에 = x_predict = x_test[:10] / (10, 3072)로 2차원 만들기
# 32 * 32 = 1024 채널 3 1024 * 3 = 3072
x_train = x_train.reshape(50000, 3072).astype('float32')/255 #.astype('float32')형변환
x_test = x_test.reshape(10000, 3072).astype('float32')/255 
x_predict = x_predict.reshape(10, 3072).astype('float32')/255

# 2차원으로 변경된 x_train도 확인해 볼거야 y_train[0]도 0을 봤으니 x_train도 [0]을 봐야 하겠지?
print("x_train data : ", x_train[0])

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(3072, ))) #(32,32,10)
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu')) # strides=2 니까 2칸씩 이동한다
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(100, activation='relu')) #Dense 디폴트는  linear
model.add(Dense(100, activation='softmax')) # 꼭 들어감

model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_Stopping = EarlyStopping(monitor='loss', patience=15, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
model.fit(x_train, y_train, epochs=500, batch_size=320, verbose=1, validation_split=0.2, callbacks=[early_Stopping])

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

# 이미 우리는 train, test로 나눠서 x,y 둘다 훈련을 시켰음
# y_predict로 변수 선언하고 model.predict를 활용해서  
y_predict = model.predict(x_predict)
y_predict_re = np.argmax(y_predict, axis=1)

print("y_real : ", y_real)
print("y_predict_re : ", y_predict_re)

