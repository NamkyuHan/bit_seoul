import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Dropout
from tensorflow.keras.layers import MaxPooling1D
from sklearn.model_selection import train_test_split

# 1. 데이터
# 데이터 로드
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 불러왔으니까 데이터 쉐이프 확인
print(x_train.shape, x_test.shape) #(50000,32,32,3), (10000,32,32,3)
print(y_train.shape, y_test.shape) #(50000,1) (10000,1)

# x_predict도 선언 했으니 쉐이프 보기!
x_predict = x_test[:10, :, :, :]
print(x_predict.shape) #(10,32,32,3)

# 스케일러 적용하기X
# 분류모델에는 스케일러 안함 게다가 이미지

# 분류니까 y값에 원핫 인코딩 설정
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

# 쉐이프 변환
# 50000, 512, 6 = 60000, 784, 1 도 됨
x_train=x_train.reshape(50000, 512, 6).astype('float32')/255. 
x_test=x_test.reshape(10000, 512, 6).astype('float32')/255.

model=Sequential()
model.add(Conv1D(10, kernel_size=2, strides=1, padding='same', input_shape=(512, 6)))
model.add(Conv1D(30, kernel_size=2,  padding='same'))
model.add(Conv1D(20, kernel_size=2, padding='same'))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(100, activation='softmax'))  

model.summary()

#3 컴파일 핏
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='loss', patience=10, mode='auto')

model.fit(x_train, y_train, epochs=30, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es])

#4. 훈련 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=128)

print('loss : ', loss)
print('accuracy : ', acc)

x_predict=x_predict.reshape(10, 512, 6).astype('float32')/255.

y_predict=model.predict(x_predict)
y_predict=np.argmax(y_predict, axis=1)
y_actually=np.argmax(y_test[:10, :], axis=1)
print('실제값 : ', y_actually)
print('예측값 : ', y_predict)

'''
loss :  4.969402313232422
accuracy :  0.125900000333786
실제값 :  [49 33 72 51 71 92 15 14 23  0]
예측값 :  [39 75 72 22 70  6 54 86 71 83]
'''
