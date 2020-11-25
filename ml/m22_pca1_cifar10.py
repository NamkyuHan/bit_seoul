# 실습 
# PCA를 통해 마구마구 0.95 이상인게 몇개?
# 1 이상
# mnist DNN과 loss / acc 를 비교

import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape) #(50000, 32, 32, 3)
# print(x_test.shape) #(10000, 32, 32, 3)

# x_train, x_test append 했으니 행이 늘어난다 그래서 60000
x = np.append(x_train, x_test, axis=0)
# print(x.shape) #(60000, 32, 32, 3)


x = x.reshape(60000, 3072).astype('float32')/255.


# PCA로 컬럼 걸러내기
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_) #누적된 합 표시

d = np.argmax(cumsum >= 0.95) +1
print("d값 : ", d) #우리가 원하는 n_components의 개수  0.95 = 217개 1 = 3072개


# pca1 변수 선언 d값을 n_components 대입, x값 트랜스폼
pca1 = PCA(n_components=d)
x = pca1.fit_transform(x)
print(x.shape) #(60000, 217)


x_train = x[:50000, :]
x_test = x[50000:, :]

#원핫 적용
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(d, ))) #(70000,154)
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu')) # strides=2 니까 2칸씩 이동한다
model.add(Dense(20, activation='relu'))
model.add(Dense(100, activation='relu')) #Dense 디폴트는  linear
model.add(Dense(10, activation='softmax')) # 꼭 들어감

model.summary()

# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
early_Stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(x_train, y_train, epochs=10000, batch_size=1000, verbose=1, validation_split=0.2, callbacks=[early_Stopping]) #원래 배치 사이즈의 디폴트는 32

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1000)

print("loss : ", loss)
print("acc : ", acc)

'''
loss :  1.6167805194854736
acc :  0.4410000145435333
'''