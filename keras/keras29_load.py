import numpy as np
from tensorflow.keras.models import load_model
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
print(x.shape) #96 4 1

# x_train = datasets[:, 0:-1]
# y_train = datasets[:, -1]
# x_test = datasets[80:]
# y_test = datasets[80:]

# print(x_train.shape) #(96, 4)
# print(y_train.shape) #(96, )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, train_size = 0.7)

print(x_train.shape)

#2. 모델구성 (불러와 보자)

model = load_model('./save/keras28.h5')
model.add(Dense(5))
model.add(Dense(1))

model.summary()

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2)

x_input = np.array([7,8,9,10]) #(4, )
x_input = x_input.reshape(1,4,1)

y_predict = model.predict(x_input)
print(y_predict)

# 그냥 불러오면 그 Dense와 이름이 같은게 있으면 되지 않아 (그래서 이름이 같다고 오류가 남)






