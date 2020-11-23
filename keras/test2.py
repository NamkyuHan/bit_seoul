'''
#1. 데이터
import numpy as np

#input
x1=np.array([range(1,101), range(711, 811), range(100)])
y1=np.array([range(101,201), range(311,411), range(100)])

#output
y2=np.array([range(4,104), range(761,861), range(100)])

x1=np.transpose(x1)
y1=np.transpose(y1)
y2=np.transpose(y2)


# print(x1.shape) #(100, 3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1,y1,y2, shuffle=True, train_size=0.7
)

# from sklearn.model_selection import train_test_split
# y3_train, y3_test = train_test_split(
#     y3, shuffle=True, train_size=0.7
# )


#2. 함수형 모델 2개 구성

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

# 모델1
input1 = Input(shape=(3, ))
dense1_1 = Dense(100, activation='relu', name='king1')(input1)
dense1_2 = Dense(70, activation='relu', name='king2')(dense1_1)
dense1_3 = Dense(5, activation='relu', name='king3')(dense1_2)
output1 = Dense(3, activation='linear', name='king4')(dense1_3)

# model1 = Model(inputs=input1, outputs=output1)

# model1.summary()

# 모델2
input2 = Input(shape=(3,))
dense2_1 = Dense(150, activation='relu', name='qeen1')(input2)
dense2_2 = Dense(110, activation='relu', name='qeen2')(dense2_1)
output2 = Dense(3, activation='linear', name='qeen3')(dense2_2) #activation='linear'인 상태

# model2 = Model(inputs=input2, outputs=output2)

# model2.summary()

#모델 병합, concatenate
from tensorflow.keras.layers import Concatenate, concatenate
# from keras.layers.merge import Concatenate, concatenate
# from keras.layers import Concatenate, concatenate

#대문자는 클래스 
merge1 = Concatenate()([output1, output2]) #2개 이상이라 list로 묶습니다
# merge1 = Concatenate()([output1, output2]) 대문자로 쓰기 위한 방법 1
# merge1 = Concatenate(axis=1)([output1, output2]) 대문자로 쓰기 위한 방법 2

# middle1 = Dense(30)(merge1)
# middle2 = Dense(7)(middle1)
# middle3 = Dense(11)(middle2)

#이름 이것도 가능 (다만, 가독성 위해 이름을 middle 1, 2, 3)
middle1 = Dense(30)(merge1)
middle1 = Dense(7)(middle1)
middle1 = Dense(11)(middle1)

################# output 모델 구성 (분기)
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)


#2 모델 정의
model = Model(inputs = [input1, input2], 
              outputs = output1)

model.summary()

#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
#훈련
model.fit([x1_train, y1_train], y2_train, epochs=100, batch_size=8,
           validation_split=0.25, verbose=1)

#4. 평가, 예측
result = model.evaluate([x1_test, y1_test], y2_test, 
                batch_size=8)


# loss = model.evaluate(x1_test, y1_test, y2_test)
# print("loss : ", loss)

y_predict = model.predict([x1_test, y1_test])
print("결과물 : ", y_predict)


#2D를 선으로 쭉 펼친 데이터 모델.
#3차원으로 구성되고 input_shape는 2차원 => LSTM과 동일
#LSTM = 연속된 데이터로 다음 데이터를 찾는 모델
#::Conv1D = 연속된 데이터로 다음 특성을 추출. 이미지와 시계열 둘 다 씀.
#LSTM은 연산량이 많아 속도가 느리므로 먼저 Conv1D로 시도하는 것도 좋은 방법...

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.layers import MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras25_split import split_x 

#1. 데이터
a=np.array(range(1,101))
size=5

#split_x 함수 
datasets=split_x(a, size)

x=datasets[:,:4]
y=datasets[:,4]

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)
x_predict=np.array([[97, 98, 99, 100]])

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_predict=scaler.transform(x_predict)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_predict=x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)


#2. 모델 구성 : Conv1D로 모델을 구성하시오
model=Sequential()
model.add(Conv1D(20, kernel_size=2, strides=1, padding='same', input_shape=(x_train.shape[1], 1)))
model.add(Conv1D(150, kernel_size=2,  padding='same'))
model.add(Conv1D(100, kernel_size=2, padding='same'))
model.add(Conv1D(80, kernel_size=2, padding='same'))
model.add(Conv1D(70, kernel_size=2, padding='same'))
model.add(Conv1D(50, kernel_size=2, padding='same'))
model.add(Conv1D(20, kernel_size=2, padding='same'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(20))
model.add(Dense(1))


#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)


#4. 예측
loss=model.evaluate(x_test, y_test, batch_size=1)

y_predict=model.predict(x_predict)

print("y_predict :", y_predict)
print("loss : ", loss)
'''