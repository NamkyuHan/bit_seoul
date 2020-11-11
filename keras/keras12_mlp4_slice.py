#1. 데이터
import numpy as np

x = np.array([range(1, 101), range(711, 811), range(100)])
y = np.array([range(101, 201), range(311, 411), range(100)])

x = np.transpose(x)
y = np.transpose(y)

#슬라이싱
x_train = x[:60]
y_train = y[:60]
x_test = x[80:]
y_test = y[80:]

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_dim=3))
# model.add(Dense(1000, input_shape=(3, )))
# (100,10,3) 만약 이런 데이터가 있다면? input_shape=(10,3) 행무시가 여기서 적용이 된다 그래서 100이 사라짐
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(3))

#3. 컴파일 시키기, 훈련시키기
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2) #train 데이터를 자를거야 0.2만큼(그럼 나머지 데이터는 validation이 됨)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

# predict 새로운 값과 원래 값을 비교하는 것(평가지표에서 평가)
y_predict = model.predict(x_test)
print("결과물 : \n : ", y_predict)

# 실습 : 결과물 오차 수정 미세조정
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))    

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : : ", r2)
