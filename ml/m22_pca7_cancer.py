# 실습 
# PCA를 통해 마구마구 0.95 이상인게 몇개?
# 1 이상
# mnist DNN과 loss / acc 를 비교

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print(x.shape) #(569, 30)
print(y.shape) #(569,)


scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)
print(x.shape) #(569, 30)


# x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]).astype('float32')/255.


# PCA로 컬럼 걸러내기
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_) #누적된 합 표시

d = np.argmax(cumsum >= 1) +1
print("d값 : ", d) 
# d? 우리가 원하는 n_components의 개수  0.95 = 10개 1 = 30개


# pca1 변수 선언 d값을 n_components 대입, x값 트랜스폼
pca1 = PCA(n_components=d)
x = pca1.fit_transform(x)
print(x.shape) 

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

#원핫 적용
# y_train=to_categorical(y_train)
# y_test=to_categorical(y_test)


#2. 모델 구성
model=Sequential()
model.add(Dense(80, activation='relu', input_shape=(d,)))
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
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['acc'])
early_Stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(x_train, y_train, epochs=10000, batch_size=1000, verbose=1, validation_split=0.2, callbacks=[early_Stopping]) #원래 배치 사이즈의 디폴트는 32

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1000)

print("loss : ", loss)
print("acc : ", acc)

# y_predict=model.predict(x_test)
# y_predict=np.argmax(y_predict, axis=1)
# y_actually=np.argmax(y_test, axis=1)
# print('실제값 : ', y_actually)
# print('예측값 : ', y_predict)


'''
loss :  0.44281649589538574
acc :  0.9473684430122375
'''