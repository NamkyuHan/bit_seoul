#1. 데이터
import numpy as np
x = np.array(range(1, 101))
y = np.array([range(101, 201), range(311, 411), range(100)])

print(x.shape) 
print(y.shape)

# x = x.reshape()
y = y.transpose()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True) # train size를 70%로 주고 내가 쓴 순서대로 잘려나간다


#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=1)) # input
model.add(Dense(5))
model.add(Dense(3)) #output

#3. 컴파일 시키기, 훈련시키기
# 하이퍼파라미터 튜닝(나의 취미이자 특기가 될것이다)
# mse 평균제곱오차 손실값은 mse로 잡을거야
# optimizer 손실 최적화를 위해서는? 이번 최적화는 아담을 쓸거야
# metrics 평가지표 평가지표는 ACC를 쓸거야 
#정리하자면? 손실값을 구할때는 평균제곱오차 mse를 쓰고 손실 최적화를 위해서 adam을 쓸거야 평가지표는 acc를 쓸거고

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# 하이퍼파라미터 튜닝(나의 취미이자 특기가 될것이다)
# fit 훈련 실행
# epochs 100번 작업할거야
# batch_size 1개씩 배치해서

# model.fit(x, y, epochs=10000, batch_size=1)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2) #train 데이터를 자를거야 0.2만큼(그럼 나머지 데이터는 validation이 됨)
# validation_data=(x_val, y_val))

#4. 평가 예측
# loss, acc = model.evaluate(x,y, batch_size=1)
# loss, acc = model.evaluate(x,y)
loss = model.evaluate(x_test, y_test)

# print("loss : ", loss)
# print("acc : ", acc)

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



