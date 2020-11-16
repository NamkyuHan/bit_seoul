from numpy import array
from keras.models import Model
from keras.layers import Dense, LSTM, Input

#1. 데이터

x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], #(13, 3)
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12], 
            [2000,3000,4000], [3000,4000,5000], [4000,5000,6000], [100,200,300]]) #(14,3)

y = array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000,400])

x_predict = array([55,65,75]) #(3, )
x_predict2 = array([6600, 6700, 6800]) 

x_predict = x_predict.reshape(1,3)

#x_train, x_predict 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_predict = scaler.transform(x_predict)

print(x)
print(x_predict)


#2. 모델구성
#이거 함수로 바꿔야함
input1 = Input(shape=(3, 1)) 
dense1 = LSTM(100, activation='relu')(input1)
dense2 = Dense(200, activation='relu')(dense1)
dense3 = Dense(300, activation='relu')(dense2)
output = Dense(1)(dense3)

model = Model(inputs=input1, outputs=output)

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es = early_Stopping = EarlyStopping(monitor='loss', patience=400, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

#3. 컴파일 시키기
model.compile(optimizer='adam', loss='mse', metrics= ['mae'])
history = model.fit(x, y, epochs=50, batch_size=1, verbose=1, validation_split=0.2, callbacks=([es, to_hist]))

x_input = array([97,98,99,100]) #(4, )
x_input = x_input.reshape(1,4,)
print(x_input.shape)

y_predict = model.predict(x_input)
print(y_predict)

#4. 평가 예측
loss, mse = model.evaluate(x, batch_size=1)
print("loss : ", loss)
print("mse : ", mse)
