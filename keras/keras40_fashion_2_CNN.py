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
'''
x_train[0] :  [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   1   0   0  13  73   0
    0   1   4   0   0   0   0   1   1   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   3   0  36 136 127  62
   54   0   0   0   1   3   4   0   0   3]
 [  0   0   0   0   0   0   0   0   0   0   0   0   6   0 102 204 176 134
  144 123  23   0   0   0   0  12  10   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0 155 236 207 178
  107 156 161 109  64  23  77 130  72  15]
 [  0   0   0   0   0   0   0   0   0   0   0   1   0  69 207 223 218 216
  216 163 127 121 122 146 141  88 172  66]
 [  0   0   0   0   0   0   0   0   0   1   1   1   0 200 232 232 233 229
  223 223 215 213 164 127 123 196 229   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0 183 225 216 223 228
  235 227 224 222 224 221 223 245 173   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0 193 228 218 213 198
  180 212 210 211 213 223 220 243 202   0]
 [  0   0   0   0   0   0   0   0   0   1   3   0  12 219 220 212 218 192
  169 227 208 218 224 212 226 197 209  52]
 [  0   0   0   0   0   0   0   0   0   0   6   0  99 244 222 220 218 203
  198 221 215 213 222 220 245 119 167  56]
 [  0   0   0   0   0   0   0   0   0   4   0   0  55 236 228 230 228 240
  232 213 218 223 234 217 217 209  92   0]
 [  0   0   1   4   6   7   2   0   0   0   0   0 237 226 217 223 222 219
  222 221 216 223 229 215 218 255  77   0]
 [  0   3   0   0   0   0   0   0   0  62 145 204 228 207 213 221 218 208
  211 218 224 223 219 215 224 244 159   0]
 [  0   0   0   0  18  44  82 107 189 228 220 222 217 226 200 205 211 230
  224 234 176 188 250 248 233 238 215   0]
 [  0  57 187 208 224 221 224 208 204 214 208 209 200 159 245 193 206 223
  255 255 221 234 221 211 220 232 246   0]
 [  3 202 228 224 221 211 211 214 205 205 205 220 240  80 150 255 229 221
  188 154 191 210 204 209 222 228 225   0]
 [ 98 233 198 210 222 229 229 234 249 220 194 215 217 241  65  73 106 117
  168 219 221 215 217 223 223 224 229  29]
 [ 75 204 212 204 193 205 211 225 216 185 197 206 198 213 240 195 227 245
  239 223 218 212 209 222 220 221 230  67]
 [ 48 203 183 194 213 197 185 190 194 192 202 214 219 221 220 236 225 216
  199 206 186 181 177 172 181 205 206 115]
 [  0 122 219 193 179 171 183 196 204 210 213 207 211 210 200 196 194 191
  195 191 198 192 176 156 167 177 210  92]
 [  0   0  74 189 212 191 175 172 175 181 185 188 189 188 193 198 204 209
  210 210 211 188 188 194 192 216 170   0]
 [  2   0   0   0  66 200 222 237 239 242 246 243 244 221 220 193 191 179
  182 182 181 176 166 168  99  58   0   0]
 [  0   0   0   0   0   0   0  40  61  44  72  41  35   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]]
y_train[0] :  9
'''
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
loss :  0.7403817176818848
acc :  0.8988000154495239
y_real :  [9 2 1 1 6 1 4 6 5 7]
y_predict_re :  [9 2 1 1 6 1 4 6 5 7]
'''