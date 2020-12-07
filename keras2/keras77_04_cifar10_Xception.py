# 배운거 토대로 최적화 튠으로 구성
# 배운거 토대로 최적화 튠으로 구성
# 배운거 토대로 최적화 튠으로 구성

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import MaxPooling2D, Flatten
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import Xception

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_predict=x_test[:10, :, :, :]

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_predict = x_predict.astype('float32')/255.

print(x_train.shape) # (50000, 32, 32, 3)
print(x_test.shape) # (10000, 32, 32, 3)

# 2. OneHotEncodeing  인코딩은 아래 코드와 같이 케라스에서 제공하는 “to_categorical()”로 쉽게 처리할 수 있습니다
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

xception = Xception(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
# trainable=False 면 더 훈련 시키지 않을거야 이미지넷 가중치 사용할거야 true면 반대
xception.trainable=False

model = Sequential()
model.add(xception)
model.add(Flatten())
model.add(Dense(30))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_Stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
model.fit(x_train, y_train, epochs=30, batch_size=20, verbose=1, validation_split=0.2, callbacks=[early_Stopping])

loss, accuracy=model.evaluate(x_test, y_test, batch_size=20)

print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict=model.predict(x_predict)
y_predict=np.argmax(y_predict, axis=1) #One hot encoding의 decoding은 numpy의 argmax를 사용한다.
y_actually=np.argmax(y_test[:10, :], axis=1)
print('실제값 : ', y_actually)
print('예측값 : ', y_predict)

'''
데이터 셋이 작아서 안됨
'''

