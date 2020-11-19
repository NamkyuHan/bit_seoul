# 다중분류
# 1. 데이터
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Dropout
from tensorflow.keras.layers import MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

dataset=load_iris()
x=dataset.data
y=dataset.target
# print(x)
# print(x.shape, y.shape) #(150, 4) (150,)

# train, test split 설정하기
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

# to_categorical 적용
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

x_train = x_train.reshape(120, 4, 1, 1)
x_test = x_test.reshape(30, 4, 1, 1)


# 2. 모델구성
# CNN 4차원

model=Sequential()
model.add(Conv2D(10, (2,2), padding='same' ,input_shape=(4, 1, 1)))
model.add(Conv2D(20, (2,2), padding='same'))
model.add(Conv2D(35, (2,2), padding='same'))
model.add(Conv2D(70, (2,2), padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(50, (2,2), padding='same'))
model.add(Conv2D(30, (2,2), padding='same'))
model.add(Flatten())
model.add(Dense(80, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

# 분류 모델은 매트릭스 주기
# 분류에서 loss는 'categorical_crossentropy'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
early_stopping=EarlyStopping(monitor='loss', patience=10, mode='auto')

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
훈련데이터 CNN
loss :  0.04041199013590813
accuracy :  1.0
실제값 :  [2 0 2 1 2 1 2 0 0 1 2 1 2 0 2 0 0 1 0 1 0 0 0 0 1 0 2 1 2 0]
예측값 :  [2 0 2 1 2 1 2 0 0 1 2 1 2 0 2 0 0 1 0 1 0 0 0 0 1 0 2 1 2 0]
'''