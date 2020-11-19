import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

#1. 데이터 로드
dataset=load_diabetes()
x=dataset.data
y=dataset.target
print(x)
print(x.shape, y.shape) #(442, 10) (442,)

# 스피릿으로 데이터 슬라이스 트레인 테스트 주기
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

#회귀 모델이고 이미지도 아니어서 스케일러 적용
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#모델에 넣어야 하니까 reshape 적용
#행무시 열우선 열 하나 늘림 442, 10, 1
x_train=x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

model=Sequential()
model.add(Conv1D(20, kernel_size=2, strides=1, padding='same', input_shape=(10, 1)))
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

model.summary()

#컴파일 핏
model.compile(loss='mse', optimizer='adam')

early_stopping=EarlyStopping(monitor='loss', patience=50, mode='min')

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2 ,callbacks=[early_stopping])

#4. 평가 예측
y_predict=model.predict(x_test)

from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score 
r2=r2_score(y_test, y_predict)
print("R2 : ", r2)

'''
RMSE :  47.70742874988129
R2 :  0.6027274860440647
'''