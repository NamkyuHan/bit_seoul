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

# MinMaxScaler() 적용하기
scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

# 스케일러 적용 후 train, test split 설정하기
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)


# 스케일링 된 x데이터 다시 확인해 보기
print(x_train.shape)#(120, 4)
print(x_test.shape)#(30, 4)

y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)

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

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1) #2행 1열 중 첫번째
plt.plot(hist.history['loss'], marker='.', c='red')
plt.plot(hist.history['val_loss'], marker='.', c='blue')
plt.grid()

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='uper right')

plt.subplot(2, 1, 2) #2행 1열 중 두번째
plt.plot(hist.history['acc'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend('acc', 'val_acc')

plt.show


'''
실습 1. test 데이터를 10개 가져와서 predict 만들것
-원핫 인코딩을 원복할 것
print('실제값 : ', y_real) 결과 : [3 4 5 2 9 1 3 9 0]
print('예측값 : ', y_predict_re) 결과 : [3 4 5 2 9 1 3 9 1]
y 값이 원핫 인코딩 되어있음
이걸 원복 시켜야 한다

실습 2. 모델 es적용 얼리스탑, 텐서보드도 넣을것
'''
