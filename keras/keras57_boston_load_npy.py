import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
'''
dataset = load_boston()
x = dataset.data
y = dataset.target
# print(x.shape, y.shape) #(506, 13) (506,)

#test_size?
# test_size: 테스트 셋 구성의 비율을 나타냅니다. 
# train_size의 옵션과 반대 관계에 있는 옵션 값이며, 주로 test_size를 지정해 줍니다. 
# 0.2는 전체 데이터 셋의 20%를 test (validation) 셋으로 지정하겠다는 의미입니다. default 값은 0.25 입니다. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
'''

x = np.load('./data/boston_x.npy')
y = np.load('./data/boston_y.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# x데이터 train, test 나눴으니 트레인 데이터 삽입 명시  
# train과 test만 나눴으니 이렇게 해야 함 
scaler = MinMaxScaler()
scaler.fit(x_train)
x = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#데이터 셋이 2차원이기 때문에 reshape 안함
model=Sequential()
model.add(Dense(80, activation='relu', input_shape=(13,)))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(350, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(700, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(480, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(280, activation='relu'))
model.add(Dense(80))
model.add(Dense(30))
model.add(Dense(1))

model.summary()
# 회귀 모델은 매트릭스 안주기
model.compile(loss='mse', optimizer='adam')
early_stopping=EarlyStopping(monitor='loss', patience=50, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2 ,callbacks=[early_stopping])

y_predict=model.predict(x_test)

from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score 
r2=r2_score(y_test, y_predict)
print("R2 : ", r2)