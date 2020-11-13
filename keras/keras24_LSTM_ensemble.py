import numpy as np
from numpy import array
#1. 데이터

x1 = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], #(13, 3)
[5,6,7], [6,7,8], [7,8,9], [8,9,10],
[9,10,11], [10,11,12], 
[20,30,40], [30,40,50], [40,50,60]])

x2 = array([[10,20,30], [20,30,40], [30,40,50], [40,50,60],
[50,60,70], [60,70,80], [70,80,90], [80,90,100],
[90,100,110], [100,110,120],
[2,3,4], [3,4,5], [4,5,6]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_predict = array([55,65,75])
x2_predict = array([65,75,85])

# x1 = x1.T
# x2 = x2.T
# y = y.T

x1 = x1.reshape(13,3,1)
x2 = x2.reshape(13,3,1)

x1_predict = x1_predict.reshape(1,3,1)
x2_predict = x2_predict.reshape(1,3,1)

print(x1.shape) #(100,3)
print(x2.shape) #(100,3)
print(y.shape) #(13, )
print(x1_predict.shape) #(3, )
print(x2_predict.shape) #(3, )




# from sklearn.model_selection import train_test_split
# x1_train, x1_test, x2_train, x2_test, y_train, y_test= train_test_split(x1, x2, y, train_size = 0.7)

from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.callbacks import EarlyStopping
# 모델 1
input1 = Input(shape=(3,1))
dense1_1= LSTM(150, activation='relu' , name='king1')(input1)
dense1_2= Dense(100, activation='relu' , name='king2')(dense1_1)
dense1_3= Dense(80, activation='relu' , name='king3')(dense1_2)
output1 = Dense(1, activation='linear' , name='king4')(dense1_3) #마지막 아웃풋은 리니어로 지정해 줘야 한다

#모델 2
input2 = Input(shape=(3,1))
dense2_1= LSTM(150, activation='relu' , name='queen1')(input2)
dense2_2= Dense(100, activation='relu' , name='queen2')(dense2_1)
dense2_3= Dense(80, activation='relu', name='queen3')(dense2_2)
output2 = Dense(1, activation='linear' , name='queen4')(dense2_3) #마지막 아웃풋은 리니어로 지정해 줘야 한다

# 모델 병합, Concatenate
from tensorflow.keras.layers import Concatenate, concatenate

merger1 = Concatenate()([output1, output2]) # 값이 2개 이상일때는 list[] 로 묶는다

# output 만들기 모델구성(분기)
output1 = Dense(200 , name='output1')(merger1)
output1 = Dense(100 , name='output1_2')(output1)
output1 = Dense(1 , name='output1_3')(output1)

# 모델 정의
model = Model(inputs=[input1, input2], outputs=output1)

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# early_Stopping = EarlyStopping(monitor='loss', patience=500, mode='auto')
model.fit([x1, x2], y, epochs=300, batch_size=1, verbose=1) #callbacks=[early_Stopping])

#4. 평가 예측
result = model.evaluate([x1, x2], y, batch_size=1)
print("result : ", result)

# x_predict = model.predict([x1, x2])
# y_predict = model.predict(y)

y_predict = model.predict([x1_predict, x2_predict])

# y_predict1 = model.predict([x1_predict, x2_predict])

print("y_predict result : ", y_predict)

# 85
