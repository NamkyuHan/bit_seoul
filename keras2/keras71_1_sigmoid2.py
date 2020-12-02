import numpy as np



#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성 linear(디폴트)는 전에 적용된 옵션이 그대로 적용되어 연산되어 진다
model = Sequential()
model.add(Dense(300, input_dim=1, activation='sigmoid'))
model.add(Dense(5000, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

#3. 컴파일 시키기, 훈련시키기


model.compile(loss='mse', optimizer='adam',metrics=['acc'])


model.fit(x, y, epochs=100, batch_size=1)

#4. 평가 예측
loss, acc = model.evaluate(x,y, batch_size=1)

print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(x)
print("결과물 : \n : ", y_pred)


