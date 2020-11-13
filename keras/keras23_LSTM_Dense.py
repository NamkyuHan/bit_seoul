import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], #(13, 3)
[5,6,7], [6,7,8], [7,8,9], [8,9,10],
[9,10,11], [10,11,12], 
[20,30,40], [30,40,50], [40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) #(13, )
x_input = np.array([50,60,70]) #(3, )

print("x.shape : ", x.shape) # x.shape :  (13, 3)
print("y.shape : ", y.shape) # y.shape :  (13,)

#shape 맞추기
#LSTM을 사용하기 위해선 reshape가 필수불가결
#dense층에선 (13, 3)을 각 1열씩이라고 판단 가능하므로 reshape 필요 x

# x = x.reshape(13, 3, 1)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,LSTM
from tensorflow.keras.callbacks import EarlyStopping #조기종료
model = Sequential()
model.add(Dense(30, activation='relu', input_dim=3)) #column 개수=3
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
early_Stopping = EarlyStopping(monitor='loss', patience=500, mode='auto')
model.fit(x, y, epochs=10000, batch_size=1, verbose=1, callbacks=[early_Stopping])

#4. 평가 예측
# result = model.predict(x_input)
# print("result : ", result)

x_input = x_input.reshape(1,3)
predict = model.predict(x_input)
print("predict :", predict)

loss = model.evaluate(x_input, np.array([80]), batch_size=1)
print("loss :", loss)

'''
표 만들기
Total params 6,661
loss 0.0077
result 79.95032

LSTM_Dense
Total params 5,581
loss 0.0077
result 79.95032
'''
