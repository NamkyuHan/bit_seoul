#OneHotEncodeing
# 1. 데이터
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_predict = x_test[:10] 내가 답을 유추해 볼건데 10개 까지 볼거야(슬라이싱)
# y_real = y_test[:10] 내가 답을 유추해 볼거면 답이 있어야 겠지? 정해져 있는 그 답도 10개 까지 볼거야(슬라이싱)
x_predict = x_test[:10]
y_real = y_test[:10]

# 프린트 해서 값을 쉐이프를 확인해보자
# x_train, x_test, y_train, y_test
# 쉐이프 확인 했으면 헷갈리지 않게 옆에다가 잘 써주자
# print(x_train[0]), print(y_train[0]) = 이건 왜? 원래 매칭되어 있는 값을 내눈으로 한번 더 볼려고
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000, ) (10000, )
print(x_train[0])
print(y_train[0])


# 데이터 전처리 1.OneHotEncodeing 
# 1. sklearn을 통해 임포트 할 수도 있다
# from sklearn.preprocessing import OneHotEncoder
# enc(변수설정) = OneHotEncoder()(OneHotEncoder 대입)
# enc.fit(Y_class) (fit 설정 x,y 데이터 train 있으면 train으로 대입)

# 2. OneHotEncodeing  인코딩은 아래 코드와 같이 케라스에서 제공하는 “to_categorical()”로 쉽게 처리할 수 있습니다
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# OneHotEncodeing으로 변경된 y_train, y_test 쉐이프 확인하기
# 대입되어 있는 y_train 값 확인하기
print("y_train shape : ", y_train.shape)
print("y_test shape : ", y_test.shape)
print("y_train data : ", y_train[0])

# CNN에 집어넣기 위해 4차원으로 x_train, x_test, x_predict reshape하기
# x_predict은 x_test를 통해 위에 10개까지 본다고 설정해 놨기 때문에 = x_predict = x_test[:10] / (10, 28, 28, 1)로 4차원 만들기 
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255 #cnn에 집어넣기 위해 4차원으로 reshape (cnn은 4차원을 받아들이기 때문에) .astype('float32')형변환
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255 #cnn에 집어넣기 위해 4차원으로 reshape (cnn은 4차원을 받아들이기 때문에) .astype('float32') 형변환
x_predict = x_predict.reshape(10, 28, 28, 1).astype('float32')/255 #cnn에 집어넣기 위해 4차원으로 reshape (cnn은 4차원을 받아들이기 때문에) .astype('float32') 형변환

# 4차원으로 변경된 x_train도 확인해 볼거야 y_train[0]도 0을 봤으니 x_train도 [0]을 봐야 하겠지?
print("x_train data : ", x_train[0])


#2. 모델 구성
from tensorflow.keras.models import Sequential #Sequential 임포트 하고
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten # 이거 네개도 사용할 거니까 임포트
import tensorflow as tf
# 분산처리 (GPU 2개 쓰는 방법)
strategy = tf.distribute.MirroredStrategy(cross_device_ops= \ 
                                          tf.distribute.HierarchicalCopyAllReduce()      
)
#cnn에 쓰는것 다 집어넣음
with strategy.scope():
model = Sequential()
#4차원으로 데이터가 들어감 행무시니까 60000날아가고 (28,28,1)이 input_shape
model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1))) # 그리고 filters가 10개니까 1개였던 필터가 10개로 바뀜(28,28,10) padding='same'이므로 그대로
model.add(Conv2D(20, (2,2), padding='valid')) #padding='valid'이므로 (28,28,10)이 (27,27,10)이 되고 filters가 20개니까 (27,27,20)으로 바뀜
model.add(Conv2D(30, (3,3))) #padding= 이 명시되어 있지 않으니 디폴트 값인 'valid' 지정 (27,27,20)에서 (25,25,20)으로 바뀜 filters가 30개니까(25,25,30)으로 바뀜
model.add(Conv2D(40, (2,2), strides=2)) # strides=2 니까 2칸씩 이동한다 #padding= 이 명시되어 있지 않으니 디폴트 값인 'valid' 지정, strides=2니까 /2 (25,25,30)에서 (12,12,30)으로 바뀜 filters가 40개니까(12,12,40)으로 바뀜
model.add(MaxPooling2D(pool_size=2)) # 사이즈를 줄이는 용도 그래서 반이 날아감 그래서 (12,12,40)이 (6,6,40)으로 바뀜
#CNN에서 컨볼루션 레이어나 맥스풀링 레이어를 반복적으로 거치면 주요 특징만 추출되고, 추출된 주요 특징은 전결합층에 전달되어 학습됩니다. 
# 컨볼루션 레이어나 맥스풀링 레이어는 주로 2차원 자료를 다루지만 전결합층에 전달하기 위해선 1차원 자료로 바꿔줘야 합니다. 
# 이 때 사용되는 것이 플래튼 레이어입니다.
model.add(Flatten()) # 그래서 (6,6,40)을 1차원으로 바꾸기 위해 6*6*40을 한다 ( ,1440)
model.add(Dense(100, activation='relu')) #Dense층으로 왔기 때문에 ( ,100)으로 바뀜
model.add(Dense(10, activation='softmax')) # activation='softmax' 꼭 들어감 Dense층으로 왔기 때문에 ( ,10)으로 바뀜 

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
# to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['acc']) # loss='categorical_crossentropy' 이거 꼭 들어감 #모든 값을 합친 것은 1이 된다 acc 때문에
              # 'mse' 쓰려면? mean_squared_error 이것도 가능
model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_Stopping]) #원래 배치 사이즈의 디폴트는 32

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

# 이미 우리는 train, test로 나눠서 x,y 둘다 훈련을 시켰음
# y_predict로 변수 선언하고 model.predict를 활용해서  
y_predict = model.predict(x_predict)
y_predict_re = np.argmax(y_predict, axis=1)

print("y_real : ", y_real)
print("y_predict_re : ", y_predict_re)


'''
실습 1. test 데이터를 10개 가져와서 predict 만들것
-원핫 인코딩을 원복할 것
print('실제값 : ', y_real) 결과 : [3 4 5 2 9 1 3 9 0]
print('예측값 : ', y_predict_re) 결과 : [3 4 5 2 9 1 3 9 1]
y 값이 원핫 인코딩 되어있음
이걸 원복 시켜야 한다

실습 2. 모델 es적용 얼리스탑, 텐서보드도 넣을것
'''
