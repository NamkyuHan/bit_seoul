import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.layers import MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras25_split import split_x # 함수 데려오기 
# 1. 데이터
# x 4개 y 1개
a = np.array(range(1, 101))
size = 5

# split_x 멋진함수 데려오고
#split_x 함수 
datasets=split_x(a, size)
print(a)

x = datasets[:, :4]
y = datasets[:, 4]

print(x.shape) #(96, 4)
print(y.shape) #(96,)

x_train, x_test, y_train, y_test= train_test_split(x, y, train_size = 0.8)
x_predict = np.array([[97, 98, 99, 100]])

# MinMaxScaler() 적용하기
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_predict = scaler.transform(x_predict)

# 인풋이 3차원 이니까!
x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_predict=x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)


#2. 모델구성 (불러와 보자)
model=Sequential()
model.add(Conv1D(20, kernel_size=2, strides=1, padding='same', input_shape=(x_train.shape[1], 1)))
model.add(Conv1D(150, kernel_size=2,  padding='same'))
model.add(Conv1D(100, kernel_size=2, padding='same'))
model.add(Conv1D(80, kernel_size=2, padding='same'))
model.add(Conv1D(70, kernel_size=2, padding='same'))
model.add(Conv1D(50, kernel_size=2, padding='same'))
model.add(Conv1D(20, kernel_size=2, padding='same'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(20))
model.add(Dense(1))

# 3. 컴파일 핏
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=30, batch_size=1, verbose=1, validation_split=0.2)

loss = model.evaluate(x_test, y_test, batch_size=1)

y_predict=model.predict(x_predict)
print("y_predict :", y_predict)
print("loss : ", loss)


# Conv1D로 모델구성하시오
# 커널사이즈 1차원으로 스트라이드 있음
# 패딩 있다
