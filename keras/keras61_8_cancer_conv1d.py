import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.utils import to_categorical

#1. 데이터 로드
dataset=load_breast_cancer()
x=dataset.data
y=dataset.target
print(x)
print(x.shape, y.shape) #(569, 30) (569,)

# 스피릿으로 데이터 슬라이스 트레인 테스트 주기
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

# 분류 모델이니까 원핫인코딩 적용? 단일 분류라 안함

# 스케일러 적용?
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#모델에 넣어야 하니까 reshape 적용
#행무시 열우선 2차원이니까 차원 하나 늘림 569, 30, 1
x_train=x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

model=Sequential()
model.add(Conv1D(20, kernel_size=2, strides=1, padding='same', input_shape=(30, 1)))
model.add(Conv1D(150, kernel_size=2,  padding='same'))
model.add(Conv1D(100, kernel_size=2, padding='same'))
model.add(Conv1D(80, kernel_size=2, padding='same'))
model.add(Conv1D(70, kernel_size=2, padding='same'))
model.add(Conv1D(50, kernel_size=2, padding='same'))
model.add(Conv1D(20, kernel_size=2, padding='same'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(20))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 컴파일 핏
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='loss', patience=50, mode='auto')

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=1)

print('loss : ', loss)
print('accuracy : ', accuracy)

'''
loss :  0.4298916757106781
accuracy :  0.9385964870452881
'''




