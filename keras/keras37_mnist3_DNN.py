#OneHotEncodeing
# 1. 데이터
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 60000, 784 컬럼
 
x_predict = x_test[:10]
y_real = y_test[:10]

print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000, ) (10000, )
print(x_train[0])
print(y_train[0])

# 결과값
# (60000, 28, 28) (10000, 28, 28)
# (60000,) (10000,)

# 10개의 그림중 내가 마음에 드는 그림을 찾아라
# 0~9가 평등하다는 조건을 갖춰야 한다
# plt.imshow(x_train[0], 'gray')
# plt.show()

# 데이터 전처리 1.OneHotEncodeing 
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)
print(y_train[0])

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_predict = x_predict.reshape(10, 784)

print(x_train[0])


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

#cnn에 쓰는것 다 집어넣음
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(784, ))) #(28,28,10)
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu')) # strides=2 니까 2칸씩 이동한다
model.add(Dense(20, activation='relu'))
model.add(Dense(100, activation='relu')) #Dense 디폴트는  linear
model.add(Dense(10, activation='softmax')) # 꼭 들어감

model.summary()

# cnn activation= 디폴트 값?
# activation : 활성화 함수 설정합니다.
# ‘linear’ : 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
# ‘relu’ : rectifier 함수, 디폴트 값 은익층에 주로 쓰입니다.
# ‘sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.
# ‘softmax’ : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard # 조기 종료, 텐서보드 임포트
early_Stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['acc']) # loss='categorical_crossentropy' 이거 꼭 들어감 #모든 값을 합친 것은 1이 된다 acc 때문에
              # 'mse' 쓰려면? mean_squared_error 이것도 가능
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_Stopping, to_hist]) #원래 배치 사이즈의 디폴트는 32

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

# 이미 우리는 train, test로 나눠서 x,y 둘다 훈련을 시켰음
# y_predict로 변수 선언하고 model.predict를 활용해서 예측한다 (x_test의 값을 가지고)  
y_predict = model.predict(x_predict)
y_predict_re = np.argmax(y_predict, axis=1)

print("y_real : ", y_real)
print("y_predict_re : ", y_predict_re)


'''
실습 1. test 데이터를 10개 가져와서 predict 만들것
-원핫 인코딩을 원복할 것
print('실제값 : ', 어쩌구 저쩌구) 결과 : [3 4 5 2 9 1 3 9 0]
print('예측값 : ', 어쩌구 저쩌구) 결과 : [3 4 5 2 9 1 3 9 1]
y 값이 원핫 인코딩 되어있음
이걸 원복 시켜야 한다

실습 2. 모델 es적용 얼리스탑, 텐서보드도 넣을것

loss :  0.2863274812698364
acc :  0.9194999933242798
y_real :  [7 2 1 0 4 1 4 9 5 9]
y_predict_re :  [7 2 1 0 4 1 4 9 5 9]
'''
