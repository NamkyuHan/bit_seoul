import numpy as np

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

x_train = datasets[:, 0:-1]
y_train = datasets[:, -1]
x_test = datasets[80:]
y_test = datasets[80:]

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

print(x_train.shape) #(96, 4)
print(y_train.shape) #(96, )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x_train, y_train, train_size = 0.7)

#LSTM 함수형 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


#이거 함수로 바꿔야함
model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(4,1))) #행무시 때문에 4 날아감
# model.add(LSTM(30, input_length=3, input_dim=1, return_sequences = True))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(optimizer='adam', loss='mse')

from tensorflow.keras.callbacks import EarlyStopping
# early_Stopping = EarlyStopping(monitor='loss', patience=100, mode='min')
early_Stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=2, validation_split=0.2)

x_input = np.array([97,98,99,100]) #(4, )
x_input = x_input.reshape(1,4,1)

y_predict = model.predict(x_input)
print(y_predict)



# 모델을 구성하시오
# fit 까지만 구성


