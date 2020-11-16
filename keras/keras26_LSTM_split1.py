import numpy as np

dataset = np.array(range(1, 11)) 
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

# x = datasets[:, 0:4]
# x = datasets[:, 4]

# x = np.reshape(x, (x.shape[0], x.shape[1],1))
# print(x.shape)

x_train = datasets[:, 0:-1]
y_train = datasets[:, -1]

print(x_train.shape) #(6, 4)
print(y_train.shape) #(6, )

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

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=300, batch_size=1, verbose=1)

x_input = np.array([7,8,9,10]) #(4, )
x_input = x_input.reshape(1,4,)

y_predict = model.predict(x_input)
print(y_predict)



# 모델을 구성하시오
# fit 까지만 구성


