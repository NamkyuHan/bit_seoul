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
from sklearn.svm import LinearSVC

dataset=load_iris()
x=dataset.data
y=dataset.target
# print(x)
# print(x.shape, y.shape) #(150, 4) (150,)

# 스케일러 적용 후 train, test split 설정하기
x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=66, test_size=0.2, shuffle=True)

# 원핫 인코딩 주석
# to_categorical 적용하기
# y_train=to_categorical(y_train) 
# y_test=to_categorical(y_test)

# MinMaxScaler() 적용하기
# scaler=MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 스케일링 된 x데이터 다시 확인해 보기
# print(x_train.shape)#(120, 4)
# print(x_test.shape)#(30, 4)

# DNN 2차원이라 reshape 필요 없음(지금 데이터 2차원)

# 2. 모델구성
# DNN 2차원

# model=Sequential()
# model.add(Dense(80, activation='relu', input_shape=(4,)))
# model.add(Dense(150, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(350, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(700, activation='relu'))
# model.add(Dense(1000, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(480, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(280, activation='relu'))
# model.add(Dense(80))
# model.add(Dense(30))
# model.add(Dense(3, activation='softmax'))

# model.summary()
model = LinearSVC()

# 회귀 모델은 매트릭스 안주기
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# early_stopping=EarlyStopping(monitor='loss', patience=50, mode='auto')
# model.fit(x_train, y_train, epochs=10000, batch_size=1, validation_split=0.2 ,callbacks=[early_stopping])
# model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.2)
model.fit(x_train, y_train)

#4. 평가, 예측
# loss, acc=model.evaluate(x_test, y_test, batch_size=1)

# print('loss : ', loss)
# print('accuracy : ', acc)

# y_predict=model.predict(x_test)
# y_predict=np.argmax(y_predict, axis=1)
# y_actually=np.argmax(y_test, axis=1)
# print('실제값 : ', y_actually)
# print('예측값 : ', y_predict)

result = model.score(x_test, y_test)
print(result)

# y_predict = model.predict(x_test)





'''
loss :  0.11547557264566422
accuracy :  0.9666666388511658
실제값 :  [2 1 1 1 0 2 1 1 2 2 1 0 1 2 0 2 0 0 0 1 2 0 1 0 0 2 0 1 1 0]
예측값 :  [2 1 1 1 0 2 2 1 2 2 1 0 1 2 0 2 0 0 0 1 2 0 1 0 0 2 0 1 1 0]
'''