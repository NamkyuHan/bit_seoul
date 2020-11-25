# 실습 
# PCA를 통해 마구마구 0.95 이상인게 몇개?
# 1 이상
# mnist DNN과 loss / acc 를 비교

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA


dataset=load_boston()
x=dataset.data
y=dataset.target
# print(x)
# print(x.shape, y.shape) #(506, 13) (506,)
scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)

#PCA로 컬럼 걸러내기
pca=PCA()
pca.fit(x)
cumsum=np.cumsum(pca.explained_variance_ratio_) #누적된 합 표시
# print(cumsum)

d=np.argmax(cumsum >= 1) + 1
# print(cumsum>=0.95) 
print(d) # 2 1



pca1=PCA(n_components=d)
x=pca1.fit_transform(x)
print(x.shape) #9 #1

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)





# x_train=x_train.reshape(x_train.shape[0], x_train.shape[1],1)
# x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)
# x_train=x_train.reshape(x_predict.shape[0], x_predict.shape[1],1)


model=Sequential()
model.add(Dense(80, activation='relu', input_shape=(x.shape[1],)))
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

model.compile(loss='mse', optimizer='adam')


early_stopping=EarlyStopping(monitor='loss', patience=50, mode='min')

model.fit(x_train, y_train, epochs=10000, batch_size=1, validation_split=0.2 ,callbacks=[early_stopping])

y_predict=model.predict(x_test)


from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score 
r2=r2_score(y_test, y_predict)
print("R2 : ", r2)

'''
RMSE :  7.888678792768543
R2 :  0.31053278584217103
'''