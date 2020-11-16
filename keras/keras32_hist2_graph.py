import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

#데이터 증가
dataset = np.array(range(1, 101)) 
size = 5

def split_x(seq, size):
    aaa = [] # 는 리스트
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset]) #for문으로 넣나 그냥 넣나 거의 동일하다
    print(type(aaa))
    return np.array(aaa)

datasets = split_x(dataset, size)
print("=================")
print(datasets)  

x = datasets[:, 0:4]
y = datasets[:, 4]

x = np.reshape(x, (x.shape[0], x.shape[1], 1))
print(x.shape) #(96, 4, 1)

# x_train = datasets[:, 0:-1]
# y_train = datasets[:, -1]
# x_test = datasets[80:]
# y_test = datasets[80:]

# print(x_train.shape) #(96, 4)
# print(y_train.shape) #(96, )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, shuffle=False, train_size = 0.9)

print(x_train.shape)

#2. 모델구성 (불러와 보자)
# model = Sequential()
# model.add(LSTM(100, input_shape=(4,1)))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))

# model = load_model('./save/keras28.h5')
# model.add(Dense(5))
# model.add(Dense(1))

# model.summary()

#LSTM 함수형 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

#이거 함수로 바꿔야함
input1 = Input(shape=(4, 1)) 
dense1 = LSTM(100, activation='relu')(input1)
dense2 = Dense(200, activation='relu')(dense1)
dense3 = Dense(300, activation='relu')(dense2)
output = Dense(1)(dense3)

model = Model(inputs=input1, outputs=output)

model.summary()

model.compile(optimizer='adam', loss='mse', metrics= ['mae'])
history = model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=1, validation_split=0.2)

x_input = np.array([97,98,99,100]) #(4, )
x_input = x_input.reshape(1,4,)
print(x_input.shape)

y_predict = model.predict(x_input)
print(y_predict)

# y_predict = model.predict(x_test)
# print(y_predict)

#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("mse : ", mse)
'''
# 모델을 구성하시오
# fit 까지만 구성

# print("==================")
# print(history)
# print("==================")
# print(history.history.keys())
# print("==================")
# print(history.history[val_loss()])
# print("==================")
# print(history.history[val_loss()])
'''
#그래프
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])

plt.title('loss % mae')
plt.ylabel('loss, mae')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train mae', 'val mae'])
plt.show()


