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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(30, activation='relu', input_shape=(3,1))) #행무시 때문에 4 날아감
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
# early_Stopping = EarlyStopping(monitor='loss', patience=100, mode='min')
early_Stopping = EarlyStopping(monitor='loss', patience=400, mode='auto')

model.fit(x, y, epochs=10000, batch_size=1, verbose=2, callbacks=[early_Stopping])

#4. 평가 예측
result = model.predict(x_input)
print("result : ", result)

'''
keras21_LSTM_earlyStopping
Total params 11,121
loss 0.6285
result 78.83546
'''