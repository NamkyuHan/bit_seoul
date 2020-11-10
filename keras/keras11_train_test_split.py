#1. 데이터
import numpy as np
x = np.array(range(1, 101)) # range는 맥스값에서 -1이다
y = np.array(range(101, 201)) # range는 맥스값에서 -1이다

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=False) # train size를 70%로 주고 내가 쓴 순서대로 잘려나간다
# 성능은 셔플을 한게 더 좋다 디폴트는 true   

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# model.add(Dense(500, input_dim=1))
model.add(Dense(500, input_shape=(1,)))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(70))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics='mae')

model.fit(x_train, y_train, epochs=100, validation_split=0.2)

loss = model.evaluate(x_test, y_test)

print("loss : ", loss)

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

print(x_test)
