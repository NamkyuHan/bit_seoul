#이미지 종류가 10개
'''
데이터 (Data)
Fashion-MNIST 데이터셋에는 10개의 카테고리가 있습니다.

레이블 설명

0 티셔츠/탑
1 바지
2 풀오버(스웨터의 일종)
3 드레스
4 코트
5 샌들
6 셔츠
7 스니커즈
8 가방
9 앵클 부츠
'''
#1. 데이터

from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# x_predict = x_test[:10] 내가 답을 유추해 볼건데 10개 까지 볼거야(슬라이싱)
# y_real = y_test[:10] 내가 답을 유추해 볼거면 답이 있어야 겠지? 정해져 있는 그 답도 10개 까지 볼거야(슬라이싱)
x_predict = x_test[:10]
y_real = y_test[:10]

# 프린트 해서 값을 쉐이프를 확인해보자
# x_train, x_test, y_train, y_test
# 쉐이프 확인 했으면 헷갈리지 않게 옆에다가 잘 써주자
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000, ) (10000, )

# 0번 값 확인
print("x_train[0] : ",x_train[0])
print("y_train[0] : ", y_train[0])

# plt.imshow(x_train[0],'gray')
# plt.show()

# 2. OneHotEncodeing  인코딩은 아래 코드와 같이 케라스에서 제공하는 “to_categorical()”로 쉽게 처리할 수 있습니다
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# OneHotEncodeing으로 변경된 y_train, y_test 쉐이프 확인하기
# 대입되어 있는 y_train 값 확인하기
print("y_train shape : ", y_train.shape) #(60000, 10)
print("y_test shape : ", y_test.shape) #(10000, 10)
print("y_train data : ", y_train[0]) #[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]

# CNN에 집어넣기 위해 4차원으로 x_train, x_test, x_predict reshape하기
# x_predict은 x_test를 통해 위에 10개까지 본다고 설정해 놨기 때문에 = x_predict = x_test[:10] / (10, 28, 28, 1)로 4차원 만들기 
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255 #cnn에 집어넣기 위해 4차원으로 reshape (cnn은 4차원을 받아들이기 때문에) .astype('float32')형변환
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255 #cnn에 집어넣기 위해 4차원으로 reshape (cnn은 4차원을 받아들이기 때문에) .astype('float32') 형변환
x_predict = x_predict.reshape(10, 28, 28, 1).astype('float32')/255 #cnn에 집어넣기 위해 4차원으로 reshape (cnn은 4차원을 받아들이기 때문에) .astype('float32') 형변환

# 4차원으로 변경된 x_train도 확인해 볼거야 y_train[0]도 0을 봤으니 x_train도 [0]을 봐야 하겠지?
print("x_train data : ", x_train[0])

#2. 모델 구성
model = Sequential()
#4차원으로 데이터가 들어감 행무시니까 60000날아가고 (28,28,1)이 input_shape
model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1)))
model.add(Conv2D(20, (2,2), padding='valid'))
model.add(Conv2D(30, (3,3)))
model.add(Conv2D(40, (2,2), strides=2))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_Stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['acc']) # loss='categorical_crossentropy' 이거 꼭 들어감 #모든 값을 합친 것은 1이 된다 acc 때문에
              # 'mse' 쓰려면? mean_squared_error 이것도 가능
hist = model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_Stopping]) #원래 배치 사이즈의 디폴트는 32

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

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1) #2행 1열 중 첫번째
plt.plot(hist.history['loss'], marker='.', c='red')
plt.plot(hist.history['val_loss'], marker='.', c='blue')
plt.grid()

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='uper right')

plt.subplot(2, 1, 2) #2행 1열 중 두번째
plt.plot(hist.history['acc'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend('acc', 'val_acc')

plt.show


'''
실습 1. test 데이터를 10개 가져와서 predict 만들것
-원핫 인코딩을 원복할 것
print('실제값 : ', y_real) 결과 : [3 4 5 2 9 1 3 9 0]
print('예측값 : ', y_predict_re) 결과 : [3 4 5 2 9 1 3 9 1]
y 값이 원핫 인코딩 되어있음
이걸 원복 시켜야 한다

실습 2. 모델 es적용 얼리스탑, 텐서보드도 넣을것
'''
