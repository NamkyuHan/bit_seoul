'''
아웃풋 2 
소프트맥스

y데이터에 대한 원핫인코딩 해야한다
분류니까 당연히 해야안다 (이진분류) 참, 거짓
카테고리 컬 엔트로피

sigmoid?
데이터를 두 개의 그룹으로 분류하는 문제에서 가장 기본적인 방법은 로지스틱 회귀분석이다. 
회귀분석과의 차이는 회귀분석에서는 우리가 원하는 것이 예측값(실수)이기 때문에 종속변수의 범위가 실수이지만 
로지스틱 회귀분석에서는 종속변수 y값이 0 또는 1을 갖는다. 
그래서 우리는 주어진 데이터를 분류할 때 0인지 1인지 예측하는 모델을 만들어야 한다.

0을 실패, 1을 성공 이라고 하겠다

1. 원 핫 = 안함
2. Dence = sigmoid
3. C.C.E = 바이너리 크로스 엔트로피
46번은 다 이진분류
아웃풋은 1 그리고 시그모이드
평가지표 매트릭스에 ACC 넣기
'''

import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Dropout
from tensorflow.keras.layers import MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
# print(x.shape, y.shape) #(569, 30) (569,)

# train, test split 설정하기
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

# MinMaxScaler() 스케일링 적용하기
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

# 스케일링 된 x데이터 다시 확인해 보기
print(x_train.shape)#(455, 30)
print(x_test.shape)#(114, 30)

# LSTM 인풋에 맞게 reshape 하기 3차원
x_train = x_train.reshape(455, 30, 1)
x_test = x_test.reshape(114, 30, 1)
# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1,)
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1,)

# 2. 모델구성
# LSTM 3차원
# 이진분류 아웃풋은1 마지막 레이어 sigmoid 적용
model=Sequential()
model.add(LSTM(80, activation='relu', input_shape=(30, 1)))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(350, activation='relu'))
model.add(Dense(700, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(280, activation='relu'))
model.add(Dense(80))
model.add(Dense(30))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 분류 모델은 매트릭스 주기
# 분류에서 loss는 'categorical_crossentropy'
# 여기서는 loss='binary_crossentropy'
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
early_stopping=EarlyStopping(monitor='loss', patience=20, mode='auto')

model.fit(x_train, y_train, epochs=10000, batch_size=1, validation_split=0.2 ,callbacks=[early_stopping])

#4. 평가, 예측
loss, acc=model.evaluate(x_test, y_test, batch_size=1)

print('loss : ', loss)
print('accuracy : ', acc)

# y_predict=model.predict(x_test)
# y_predict=np.argmax(y_predict)
# y_actually=np.argmax(y_test)
# print('실제값 : ', y_actually)
# print('예측값 : ', y_predict)

'''
loss :  0.13504889607429504
accuracy :  0.9649122953414917
'''




