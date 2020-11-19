#OneHotEncodeing
# 1. 데이터
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_predict = x_test[:10] 내가 답을 유추해 볼건데 10개 까지 볼거야(슬라이싱)
# y_real = y_test[:10] 내가 답을 유추해 볼거면 답이 있어야 겠지? 정해져 있는 그 답도 10개 까지 볼거야(슬라이싱)
x_predict=x_test[:10, :, :]


# 프린트 해서 값을 쉐이프를 확인해보자
# x_train, x_test, y_train, y_test
# 쉐이프 확인 했으면 헷갈리지 않게 옆에다가 잘 써주자
# print(x_train[0]), print(y_train[0]) = 이건 왜? 원래 매칭되어 있는 값을 내눈으로 한번 더 볼려고
# print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape) #(60000, ) (10000, )
# print(x_train[0])
# print(y_train[0])

# np.save('./data/mnist_x_train.npy', arr=x_train)
# np.save('./data/mnist_x_test.npy', arr=x_test)
# np.save('./data/mnist_y_train.npy', arr=y_train)
# np.save('./data/mnist_y_test.npy', arr=y_test)

x_train = np.load('./data/mnist_x_train.npy')
x_test = np.load('./data/mnist_x_test.npy')
y_train = np.load('./data/mnist_y_train.npy')
y_test = np.load('./data/mnist_y_test.npy')

# print(x_train[0]), print(y_train[0]) = 이건 왜? 원래 매칭되어 있는 값을 내눈으로 한번 더 볼려고
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000, ) (10000, )
# print(x_train[0])
# print(y_train[0])



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


# 4차원으로 변경된 x_train도 확인해 볼거야 y_train[0]도 0을 봤으니 x_train도 [0]을 봐야 하겠지?
print("x_train data : ", x_train[0])


# #2. 모델 구성
# from tensorflow.keras.models import Sequential #Sequential 임포트 하고
# from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten # 이거 네개도 사용할 거니까 임포트


# from tensorflow.keras.models import load_model
# model = load_model('./save/model_test02_2.h5')
# model.summary()

# 3. 컴파일 훈련

from tensorflow.keras.models import load_model
model = load_model('./model/mnist-01-0.066135.hdf5')


# 4. 평가 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])

# 이미 우리는 train, test로 나눠서 x,y 둘다 훈련을 시켰음
# y_predict로 변수 선언하고 model.predict를 활용해서

x_predict=x_predict.reshape(10, 28, 28,1).astype('float32')/255.

y_predict = model.predict(x_predict)
y_predict = np.argmax(y_predict, axis=1)
y_actually=np.argmax(y_test[:10, :], axis=1)

print('실제값 : ', y_actually)
print('예측값 : ', y_predict)



'''
loss :  0.07218430936336517
acc :  0.9865000247955322
실제값 :  [7 2 1 0 4 1 4 9 5 9]
예측값 :  [7 2 1 0 4 1 4 9 5 9]
'''

'''
로드한 모델로 훈련한 값
loss :  0.07251544296741486
acc :  0.9843999743461609
실제값 :  [7 2 1 0 4 1 4 9 5 9]
예측값 :  [7 2 1 0 4 1 4 9 5 9]
'''
'''
컴파일 핏 주석처리 하고 로드한 모델로 돌린 값
loss :  0.07218430936336517
acc :  0.9865000247955322
실제값 :  [7 2 1 0 4 1 4 9 5 9]
예측값 :  [7 2 1 0 4 1 4 9 5 9]
'''
'''
핏 다음에 세이브를 하고 로드를 하면 내가 처음 했던 가중치까지 저장되어
훈련시 항상 일정한 결과값이 나오게 된다
'''
