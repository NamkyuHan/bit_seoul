
#이미지 종류가 10개
#1. 데이터
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# x_predict = x_test[:10] 내가 답을 유추해 볼건데 10개 까지 볼거야(슬라이싱)
# y_real = y_test[:10] 내가 답을 유추해 볼거면 답이 있어야 겠지? 정해져 있는 그 답도 10개 까지 볼거야(슬라이싱)
x_predict = x_test[:10]
y_real = y_test[:10]

# 프린트 해서 값을 쉐이프를 확인해보자
# x_train, x_test, y_train, y_test
# 쉐이프 확인 했으면 헷갈리지 않게 옆에다가 잘 써주자
# print(x_train[0]), print(y_train[0]) = 이건 왜? 원래 매칭되어 있는 값을 내눈으로 한번 더 볼려고
print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)
print("x_train[0] : ",x_train[0])
'''
[[[ 59  62  63]        
  [ 43  46  45]
  [ 50  48  43]
  ...
  [158 132 108]
  [152 125 102]
  [148 124 103]]

 [[ 16  20  20]
  [  0   0   0]
  [ 18   8   0]
  ...
  [123  88  55]
  [119  83  50]
  [122  87  57]]

 [[ 25  24  21]
  [ 16   7   0]
  [ 49  27   8]
  ...
  [118  84  50]
  [120  84  50]
  [109  73  42]]

 ...

 [[208 170  96]
  [201 153  34]
  [198 161  26]
  ...
  [160 133  70]
  [ 56  31   7]
  [ 53  34  20]]

 [[180 139  96]
  [173 123  42]
  [186 144  30]
  [184 148  94]
  [ 97  62  34]
  [ 83  53  34]]

 [[177 144 116]
  [168 129  94]
  [179 142  87]
  ...
  [216 184 140]
  [151 118  84]
  [123  92  72]]]
'''
print("y_train[0] : ", y_train[0]) #[6]
# plt.imshow(x_train[0])
# plt.show()

# 2. OneHotEncodeing  인코딩은 아래 코드와 같이 케라스에서 제공하는 “to_categorical()”로 쉽게 처리할 수 있습니다
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# OneHotEncodeing으로 변경된 y_train, y_test 쉐이프 확인하기
# 대입되어 있는 y_train 값 확인하기
print("y_train shape : ", y_train.shape) #(50000, 10)
print("y_test shape : ", y_test.shape) #(10000, 10)
print("y_train data : ", y_train[0]) #[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]

# CNN에 집어넣기 위해 4차원으로 x_train, x_test, x_predict reshape하기
# x_predict은 x_test를 통해 위에 10개까지 본다고 설정해 놨기 때문에 = x_predict = x_test[:10] / (10, 28, 28, 1)로 4차원 만들기 
# x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255 #cnn에 집어넣기 위해 4차원으로 reshape (cnn은 4차원을 받아들이기 때문에) .astype('float32')형변환
# x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255 #cnn에 집어넣기 위해 4차원으로 reshape (cnn은 4차원을 받아들이기 때문에) .astype('float32') 형변환
# x_predict = x_predict.reshape(10, 28, 28, 1).astype('float32')/255 #cnn에 집어넣기 위해 4차원으로 reshape (cnn은 4차원을 받아들이기 때문에) .astype('float32') 형변환

# 4차원으로 변경된 x_train도 확인해 볼거야 y_train[0]도 0을 봤으니 x_train도 [0]을 봐야 하겠지?
print("x_train data : ", x_train[0])

model = Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=(32,32,3)))
model.add(Conv2D(20, (2,2), padding='valid'))
model.add(Conv2D(30, (3,3)))
model.add(Conv2D(40, (2,2), strides=2))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_Stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_Stopping])

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
loss :  1.282124400138855
acc :  0.5408999919891357
y_real :  [[3][8][8][0][6][6][1][6][3][1]]
y_predict_re :  [3 8 1 0 6 6 3 6 3 1]
'''











