# 1. 데이터
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) #(4,3)
y = np.array([4,5,6,7])                            #(4, )

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)
# x = x.reshape(4,3,1)

print("x.shape : ", x.shape)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(30, activation='relu', input_shape=(3,1))) #행무시 때문에 4 날아감
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(1))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=200, batch_size=1, verbose=2)

x_input = np.array([5,6,7]) #(3, ) -> (1, 3, 1)
x_input = x_input.reshape(1,3,1)

#4. 평가 예측
# result = model.evaluate(x, y, batch_size=1)
# print("result : ", result)

result = model.predict(x_input)
print("result : ", result)

'''
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y1_test, y_predict))  

from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y_predict)
print("R2 : ", r2) 
'''
