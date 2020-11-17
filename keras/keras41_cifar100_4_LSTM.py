from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import MaxPooling2D, Flatten
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_predict=x_test[:10, :, :]

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

# print(y_train.shape, y_test.shape) #(60000,10), (10000,10)
# print(y_train[0])

#MinMaxScaler의 효과를 주는 형변환
x_train=x_train.reshape(60000,28*7, 4).astype('float32')/255. #(60000, 28x28)
x_test=x_test.reshape(10000,28*7, 4).astype('float32')/255.
x_predict=x_predict.reshape(10, 28*7, 4).astype('float32')/255.


model=Sequential()
model.add(LSTM(20, activation='relu', input_shape=(28*7, 4)))
model.add(Dense(6000, activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dense(1500, activation='relu'))
model.add(Dense(900, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(140, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(10, activation='softmax')) 

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es=EarlyStopping(monitor='loss', patience=10, mode='auto')
# to_hist=TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=10, batch_size=1000, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=1000)

print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict=model.predict(x_predict)
y_predict=np.argmax(y_predict, axis=1)
y_actually=np.argmax(y_test[:10, :], axis=1)
print('실제값 : ', y_actually)
print('예측값 : ', y_predict)

''' 좀 더 조정해보자
loss :  1.1079400777816772
accuracy :  0.5651000142097473
실제값 :  [9 2 1 1 6 1 4 6 5 7]
예측값 :  [9 2 1 1 6 1 3 2 7 7]
'''