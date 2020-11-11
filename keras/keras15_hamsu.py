#1. 데이터
import numpy as np
x = np.array([range(1, 101), range(711,811), range(100)])
y = np.array(range(101, 201))


# [] 이걸로 바꾸면 일단 3,100으로 나온다

# x = np.transpose(x)
# x = x.transpose()
# 둘다 가능
# y = np.transpose(y)

# print(x[1][10])
# print(x[10][1])
print(x)
print(x.shape) #(3, )
# (100,3)

x = np.transpose(x)
y = np.transpose(y)

print(x.shape) #(3,100)

# y1, y2, y3 = W1X1 + W2X2 + W3X3 + b


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=False) # train size를 70%로 주고 내가 쓴 순서대로 잘려나간다


#2. 모델구성
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input

'''
model = Sequential()
model.add(Dense(5, input_shape=(3, ), activation='relu')) #activation='linear' 댄스층에는 기본값이 linear
# (100,10,3) 만약 이런 데이터가 있다면? input_shape=(10,3) 행무시가 여기서 적용이 된다 그래서 100이 사라짐
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))
'''

# 각자 꼬리를 따라서 들어간다
input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu')(input1) #활성화 함수 모든 레이어마다 활성화 함수가 들어가 있다 디폴트 값도 존재함 relu 쓰면 평타 85%이상
dense2 = Dense(4, activation='relu')(dense1)
dense3 = Dense(3, activation='relu')(dense2)
output1 = Dense(1)(dense3) #마지막 아웃풋은 리니어로 지정해 줘야 한다
model = Model(inputs=input1, outputs=output1) #마지막에 모델을 지정해 줘야 한다 (처음 인풋과 마지막 아웃풋을 지정해 줘야 한다)


model.summary()


'''
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
model.fit(x_train, y_train, epochs=100, validation_split=0.2, verbose=1)

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
'''


