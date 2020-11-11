#1. 데이터 2개에서 3개
import numpy as np
x1 = np.array([range(1, 101), range(711,811), range(100)])
x2 = np.array([range(4, 104), range(761,861), range(100)])

y1 = np.array([range(101, 201), range(311,411), range(100)])

x1 = x1.T
x2 = x2.T
y1 = y1.T

# x1 = np.transpose(x1)
# x2 = np.transpose(x2)
# y1 = np.transpose(y1)
# y2 = np.transpose(y2)
# y3 = np.transpose(y3)

# print(x1.shape) #(100,3)
# print(x2.shape) #(100,3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test= train_test_split(x1, x2, y1, train_size = 0.7)

# y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(y1, y2, y3, shuffle=True, train_size=0.7)

# from sklearn.model_selection import train_test_split
# y3_train, y3_test = train_test_split(y3, shuffle=True, train_size=0.7)

from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input

'''
# 첫번째 모델
# 각자 꼬리를 따라서 들어간다
input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu')(input1) #활성화 함수 모든 레이어마다 활성화 함수가 들어가 있다 디폴트 값도 존재함 relu 쓰면 평타 85%이상
dense2 = Dense(4, activation='relu')(dense1)
dense3 = Dense(3, activation='relu')(dense2)
output1 = Dense(1)(dense3) #마지막 아웃풋은 리니어로 지정해 줘야 한다
model1 = Model(inputs=input1, outputs=output1) #마지막에 모델을 지정해 줘야 한다 (처음 인풋과 마지막 아웃풋을 지정해 줘야 한다)

model1.summary()

# 두번째 모델
# 각자 꼬리를 따라서 들어간다
input2 = Input(shape=(3,))
dense4 = Dense(5, activation='relu')(input2) #활성화 함수 모든 레이어마다 활성화 함수가 들어가 있다 디폴트 값도 존재함 relu 쓰면 평타 85%이상
dense5 = Dense(4, activation='relu')(dense4)
dense6 = Dense(3, activation='relu')(dense5)
output2 = Dense(1)(dense6) #마지막 아웃풋은 리니어로 지정해 줘야 한다
model2 = Model(inputs=input2, outputs=output2) #마지막에 모델을 지정해 줘야 한다 (처음 인풋과 마지막 아웃풋을 지정해 줘야 한다)

model2.summary()
'''

# 모델 1
input1 = Input(shape=(3, ))
dense1_1= Dense(10, activation='relu' , name='king1')(input1)
dense1_2= Dense(10, activation='relu' , name='king2')(dense1_1)
dense1_3= Dense(10, activation='relu' , name='king3')(dense1_2)
output1 = Dense(3, activation='linear' , name='king4')(dense1_3) #마지막 아웃풋은 리니어로 지정해 줘야 한다

#모델 2
input2 = Input(shape=(3, ))
dense2_1= Dense(10, activation='relu' , name='queen1')(input2)
dense2_2= Dense(10, activation='relu' , name='queen2')(dense2_1)
dense2_3= Dense(10, activation='relu', name='queen3')(dense2_2)
output2 = Dense(3, activation='linear' , name='queen4')(dense2_3) #마지막 아웃풋은 리니어로 지정해 줘야 한다


# 모델 병합, Concatenate
from tensorflow.keras.layers import Concatenate, concatenate
# from keras.layers.merge import concatenate, concatenate
# from keras.layers import Concatenate, concatenate

# merger1 = concatenate([output1, output2]) # 값이 2개 이상일때는 list[] 로 묶는다
merger1 = Concatenate()([output1, output2]) # 값이 2개 이상일때는 list[] 로 묶는다
# merger1 = Concatenate([output1, output2]) # 값이 2개 이상일때는 list[] 로 묶는다
# merger1 = Concatenate()([output1, output2]) # 값이 2개 이상일때는 list[] 로 묶는다


middle1 = Dense(30 , name='middle1')(merger1)
middle2 = Dense(7 , name='middle2')(middle1)
middle3 = Dense(11 , name='middle3')(middle2) ##--

# output 만들기 모델구성(분기)
output1 = Dense(30 , name='output1')(middle1)
output1 = Dense(7 , name='output1_2')(output1)
output1 = Dense(3 , name='output1_3')(output1)

# 모델 정의
model = Model(inputs=[input1, input2], outputs=output1)

# model.summary()

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], y1_train, epochs=100, batch_size=1, validation_split=0.25, verbose=1)

#4. 평가 예측
result = model.evaluate([x1_test, x2_test], y1_test, batch_size=1)
print("result : ", result)

y_predict = model.predict([x1_test, x2_test])

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y1_test, y_predict))  

from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y_predict)
print("R2 : ", r2) 
