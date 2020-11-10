# 실습
# R2를 음수가 아닌 0.5 이하로 줄이기
# 레이어는 인풋과 아웃풋을 포함 7개 이상(히든이 5개 이상)
# 히든레이어 노드는 레이어당 각각 최소 10개 이상
# batch-size = 1
# epochs = 100이상
# 데이터 조작 금지

import numpy as np



#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델구성
# 하이퍼파라미터 튜닝(나의 취미이자 특기가 될것이다)
# Sequential을 통해 (연산) 을 만든다
# layers를 통해 레이어(줄,층) 을 만든다
# Dense DNN
# Dense 레이어는 입력과 출력을 모두 연결해주며 입력과 출력을 각각 연결해주는 가중치를 포함하고 있습니다. 
# 입력이 3개 출력이 4개라면 가중치는 총 3X4인 12개가 존재하게 됩니다. 
# Dense레이어는 머신러닝의 기본층으로 영상이나 서로 연속적으로 상관관계가 있는 데이터가 아니라면 
# Dense레이어를 통해 학습시킬 수 있는 데이터가 많다는 뜻이 됩니다. 

model = Sequential()
model.add(Dense(10000, input_dim=1))
# input_dim = 1, 입력 차원이 1이라는 뜻이며 입력 노드가 한개라고 생각하면 됩니다.
# 만약 x배열의 데이터가 2개라면 2, 3개라면 3으로 지정을 해줍니다.
# 그 다음, 만든 시퀀스 오브젝트 model에 5개의 노드를 Dense레이어를 통해 연결해줍니다. 여기서 add를 통해 하나의 레이어를 추가해주는 것입니다. 
# input 1 output 5
model.add(Dense(10000))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10000))
model.add(Dense(100))
model.add(Dense(5000))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

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
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가 예측
# loss, acc = model.evaluate(x,y, batch_size=1)
# loss, acc = model.evaluate(x,y)
loss = model.evaluate(x_test,y_test)

print("loss : ", loss)
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
