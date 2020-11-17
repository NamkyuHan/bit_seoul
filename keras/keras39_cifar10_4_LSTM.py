#이미지 종류가 10개
#1. 데이터
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#빈칸 그대로 두지 말고 원래 shape처럼 공백 : 넣어주기!
# x_predict = x_test[:10, :, :] 내가 답을 유추해 볼건데 10개 까지 볼거야(슬라이싱)
# y_real = y_test[:10, :, :] 내가 답을 유추해 볼거면 답이 있어야 겠지? 정해져 있는 그 답도 10개 까지 볼거야(슬라이싱)
x_predict = x_test[:10, :, :, :]

# 프린트 해서 값을 쉐이프를 확인해보자
# x_train, x_test, y_train, y_test
# 쉐이프 확인 했으면 헷갈리지 않게 옆에다가 잘 써주자
# print(x_train[0]), print(y_train[0]) = 이건 왜? 원래 매칭되어 있는 값을 내눈으로 한번 더 볼려고
print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)
print("x_train[0] : ", x_train[0])
print("y_train[0] : ", y_train[0]) #[6]

# plt.imshow(x_train[0])
# plt.show()

#다중 분류 데이터 전처리 (1).OneHotEncoding
# 1. sklearn을 통해 임포트 할 수도 있다
# from sklearn.preprocessing import OneHotEncoder
# enc(변수설정) = OneHotEncoder()(OneHotEncoder 대입)
# enc.fit(Y_class) (fit 설정 x,y 데이터 train 있으면 train으로 대입)
# 2. OneHotEncodeing  인코딩은 아래 코드와 같이 케라스에서 제공하는 “to_categorical()”로 쉽게 처리할 수 있습니다
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# OneHotEncodeing으로 변경된 y_train, y_test 쉐이프 확인하기
# 대입되어 있는 y_train 값 확인하기
print("y_train shape : ", y_train.shape) #(50000, 10)
print("y_test shape : ", y_test.shape) #(10000, 10)
print("y_train data : ", y_train[0]) #[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]


# LSTM에 집어넣기 위해 3차원으로 x_train, x_test, x_predict reshape하기
# x_predict은 x_test를 통해 위에 10개까지 본다고 설정해 놨기 때문에 = x_predict = x_test[:10, :, :, :] / (10, 1024, 3)로 3차원 만들기 
x_train = x_train.reshape(50000, 1024, 3).astype('float32')/255.
x_test = x_test.reshape(10000, 1024, 3).astype('float32')/255.
# x_train, x_test를 reshape 했기 때문에 x_predict도 동일하게 형태를 바꿔준다
x_predict = x_predict.reshape(10, 1024, 3).astype('float32')/255.

# 4차원으로 변경된 x_train도 확인해 볼거야 y_train[0]도 0을 봤으니 x_train도 [0]을 봐야 하겠지?
# print("x_train data : ", x_train[0])

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1024, 3))) #(32,32,10)
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu')) # strides=2 니까 2칸씩 이동한다
model.add(Dense(20, activation='relu'))
model.add(Dense(100, activation='relu')) #Dense 디폴트는  linear
model.add(Dense(10, activation='softmax')) # 꼭 들어감

model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_Stopping = EarlyStopping(monitor='loss', patience=15, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
model.fit(x_train, y_train, epochs=1, batch_size=2000, verbose=1, validation_split=0.2, callbacks=[early_Stopping])

#4. 평가 예측
# model.evaluate를 통해 결과값을 예측해보자 여기에는 테스트 값만 들어가야 함
loss, acc = model.evaluate(x_test, y_test, batch_size=2000)
print("loss : ", loss)
print("acc : ", acc)

# 이미 우리는 train, test로 나눠서 x,y 둘다 훈련을 시켰음
# x_train, x_test를 reshape 했기 때문에 x_predict도 동일하게 형태를 바꿔준다
print("x_predict : ", x_predict)

# 훈련을 통해 나온 x_predict 값을 y_predict에 넣자
y_predict = model.predict(x_predict)
# y_train, y_test를 OneHotEncoding을 했으니 데이터 복호화 진행
y_predict = np.argmax(y_predict, axis=1)
# 가장 최근의 y_test 사이즈를 확인한 후 복호화 진행
y_real = np.argmax(y_test[:10, :], axis=1)
print('실제값(y_real) : ', y_real)
print('예측값(y_predict) : ', y_predict)

'''
loss :  1.7383522987365723
acc :  0.374099999666214
y_real :  [[3][8][8][0][6][6][1][6][3][1]]
y_predict_re :  [3 8 8 8 2 6 3 6 5 1]
'''
