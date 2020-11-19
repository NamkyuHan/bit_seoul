# 다중분류
# 1. 데이터
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

dataset=load_iris()
x=dataset.data
y=dataset.target
# print(x)
# print(x.shape, y.shape) #(150, 4) (150,)

# train, test split 설정하기
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

# to_categorical 적용하기
y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)

# MinMaxScaler() 적용하기
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 스케일링 된 x데이터 다시 확인해 보기
print(x_train.shape)#(120, 4)
print(x_test.shape)#(30, 4)


# 이렇게도 쓸 수 있음
# x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
# x_test=x_test.reshape(x_test.shape[0],x_test.shape[1], 1, 1)

x_train = x_train.reshape(120, 4, 1)
x_test = x_test.reshape(30, 4, 1)

# 2. 모델구성
# LSTM 3차원

model=Sequential()
model.add(LSTM(80, activation='relu', input_shape=(4, 1)))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(350, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(700, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(480, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(280, activation='relu'))
model.add(Dense(80))
model.add(Dense(30))
model.add(Dense(3, activation='softmax'))

model.summary()

# 회귀 모델은 매트릭스 안주기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
early_stopping=EarlyStopping(monitor='loss', patience=50, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2 ,callbacks=[early_stopping])

#4. 평가, 예측
loss, acc=model.evaluate(x_test, y_test, batch_size=32)

print('loss : ', loss)
print('accuracy : ', acc)

y_predict=model.predict(x_test)
y_predict=np.argmax(y_predict, axis=1)
y_actually=np.argmax(y_test, axis=1)
print('실제값 : ', y_actually)
print('예측값 : ', y_predict)

'''
loss :  0.07604072988033295
accuracy :  0.9666666388511658
실제값 :  [2 1 0 0 1 1 1 0 2 1 0 1 2 1 2 0 1 1 0 2 0 2 2 1 1 0 1 1 0 2]
예측값 :  [2 1 0 0 1 1 1 0 2 1 0 1 2 1 2 0 1 1 0 1 0 2 2 1 1 0 1 1 0 2]
'''



