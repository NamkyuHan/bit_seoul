import numpy as np



#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3 컴파일
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
optimizer = Adam(lr=0.001) # loss :  0.11590605974197388 결과물 :  [[10.551586]]
# optimizer = Adadelta(lr=0.001) # loss :  107.45548248291016 결과물 :  [[-7.383636]]
# optimizer = Adamax(lr=0.001) # loss :  0.04750935733318329 결과물 :  [[10.704584]]
# optimizer = Adagrad(lr=0.001) # loss :  3.2570152282714844 결과물 :  [[7.6069584]]
# optimizer = RMSprop(lr=0.001) # loss :  5.2008348575327545e-05 결과물 :  [[10.987425]]
# optimizer = SGD(lr=0.001) # loss :  2.758640448519145e-06 결과물 :  [[10.997032]]
# optimizer = Nadam(lr=0.001) # loss :  7.499068206684445e-12 결과물 :  [[10.999997]]


model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])


model.fit(x, y, epochs=100, batch_size=1)

#4. 평가 예측
loss, mse = model.evaluate(x,y, batch_size=1)

# print("loss : ", loss)

y_pred = model.predict([11])
print("loss : ", loss, "결과물 : ", y_pred)


