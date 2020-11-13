import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], #(13, 3)
[5,6,7], [6,7,8], [7,8,9], [8,9,10],
[9,10,11], [10,11,12], 
[20,30,40], [30,40,50], [40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) #(13, )

x_input = np.array([50,60,70]) #(3, )
# 실습 LSTM 완성하시오
# 예측값

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)
print("x_input.shape : ", x_input.shape)

x = x.reshape(13,3,1)
x_input = x_input.reshape(1,3,1)
print("x.shape : ", x.shape)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

# model = Sequential()
# model.add(LSTM(30, activation='relu', input_shape=(3,1))) #행무시 때문에 4 날아감
# model.add(Dense(30))
# model.add(Dense(30))
# model.add(Dense(30))
# model.add(Dense(1))

# model.summary()

# 각자 꼬리를 따라서 들어간다
input1 = Input(shape=(3,1))
dense1 = LSTM(300, activation='relu')(input1) #활성화 함수 모든 레이어마다 활성화 함수가 들어가 있다 디폴트 값도 존재함 relu 쓰면 평타 85%이상
dense2 = Dense(300, activation='relu')(dense1)
dense3 = Dense(300, activation='relu')(dense2)
output1 = Dense(1)(dense3) #마지막 아웃풋은 리니어로 지정해 줘야 한다
model = Model(inputs=input1, outputs=output1) #마지막에 모델을 지정해 줘야 한다 (처음 인풋과 마지막 아웃풋을 지정해 줘야 한다)

# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=500, batch_size=1, verbose=2)

#4. 평가 예측
# result = model.predict(x_input)
# print("result : ", result)

y_predict = model.predict(x_input)
print(y_predict)


'''
표 만들기
Total params 6,661
loss 0.0077
result 79.95032
'''
