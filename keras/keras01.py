import numpy as np



#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델구성
# 하이퍼파라미터 튜닝(나의 취미이자 특기가 될것이다)
# Sequential을 통해 (연산) 을 만든다
# layers를 통해 레이어(줄,층) 을 만든다
# Dense DNN 
model = Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))

#3. 컴파일 시키기, 훈련시키기
# 하이퍼파라미터 튜닝(나의 취미이자 특기가 될것이다)
# mse 평균제곱오차 손실값은 mse로 잡을거야
# optimizer 손실 최적화를 위해서는? 이번 최적화는 아담을 쓸거야
# metrics 평가지표 평가지표는 ACC를 쓸거야 
#정리하자면? 손실값을 구할때는 평균제곱오차 mse를 쓰고 손실 최적화를 위해서 adam을 쓸거야 평가지표는 acc를 쓸거고

model.compile(loss='mse', optimizer='adam',metrics=['acc'])

# 하이퍼파라미터 튜닝(나의 취미이자 특기가 될것이다)
# fit 훈련 실행
# epochs 100번 작업할거야
# batch_size 1개씩 배치해서
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가 예측
loss, acc = model.evaluate(x,y, batch_size=1)

print("loss : ", loss)
print("acc : ", acc)
