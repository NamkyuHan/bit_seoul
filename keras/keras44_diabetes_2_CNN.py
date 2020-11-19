import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

dataset=load_diabetes()
x=dataset.data
y=dataset.target
# print(x)
# print(x.shape, y.shape) #(442, 10) (442,)



x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)


scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1], 1, 1)


model=Sequential()
model.add(Conv2D(512, (2,2), padding='same' ,input_shape=(10, 1, 1)))
model.add(Conv2D(256, (2,2), padding='same'))
model.add(Conv2D(64, (1,1), padding='same'))
model.add(Conv2D(32, (1,1), padding='same'))
model.add(Flatten())
model.add(Dense(4, activation='relu'))
model.add(Dense(1))


model.summary()

model.compile(loss='mse', optimizer='adam')


early_stopping=EarlyStopping(monitor='loss', patience=50, mode='min')

model.fit(x_train, y_train, epochs=500, batch_size=1, validation_split=0.2 ,callbacks=[early_stopping])

y_predict=model.predict(x_test)


from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score 
r2=r2_score(y_test, y_predict)
print("R2 : ", r2)

'''
RMSE :  51.99699243224366
R2 :  0.41962224968545947
안좋아
'''
'''
import numpy as np
from sklearn.datasets import load_diabetes
dataset = load_diabetes()

x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True) # train size를 70%로 주고 내가 쓴 순서대로 잘려나간다
# 성능은 셔플을 한게 더 좋다 디폴트는 true   

x_predict = x_test[:10]
y_real = y_test[:10]

#x_train, x_predict 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_predict = scaler.transform(x_predict)

print("x_data : ", x)
print("y_target : ", y)
print(x.shape) #(442, 10)
print(y.shape) #(442,)
print(x_train[0])

# [  9.82349   0.       18.1       0.        0.671     6.794    98.8
#    1.358    24.      666.       20.2     396.9      21.24   ]

print(y_train[0]) #13.3
'''