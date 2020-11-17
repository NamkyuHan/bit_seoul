'''
x
506 행 13 열 
CRIM     per capita crime rate by town
ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS    proportion of non-retail business acres per town
CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX      nitric oxides concentration (parts per 10 million)
RM       average number of rooms per dwelling
AGE      proportion of owner-occupied units built prior to 1940
DIS      weighted distances to five Boston employment centres
RAD      index of accessibility to radial highways
TAX      full-value property-tax rate per $10,000
PTRATIO  pupil-teacher ratio by town
B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT    % lower status of the population

y
506 행 1 열
target (MEDV)     Median value of owner-occupied homes in $1000's

[01]  CRIM 자치시(town) 별 1인당 범죄율  
[02]  ZN 25,000 평방피트를 초과하는 거주지역의 비율 
[03]  INDUS 비소매상업지역이 점유하고 있는 토지의 비율 
[04]  CHAS 찰스강에 대한 더미변수(강의 경계에 위치한 경우는 1, 아니면 0) 
[05]  NOX 10ppm 당 농축 일산화질소  
[06]  RM 주택 1가구당 평균 방의 개수 
[07]  AGE 1940년 이전에 건축된 소유주택의 비율 
[08]  DIS 5개의 보스턴 직업센터까지의 접근성 지수 
[09]  RAD 방사형 도로까지의 접근성 지수 
[10]  TAX 10,000 달러 당 재산세율 
[11]  PTRATIO 자치시(town)별 학생/교사 비율
[12]  B 1000(Bk-0.63)^2, 여기서 Bk는 자치시별 흑인의 비율을 말함. 
[13]  LSTAT 모집단의 하위계층의 비율(%)  
[14]  MEDV 본인 소유의 주택가격(중앙값) (단위: $1,000)
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target
# print(x.shape, y.shape) #(506, 13) (506,)

# x데이터 train, test 나눴으니 트레인 데이터 삽입 명시  
# train과 test만 나눴으니 이렇게 해야 함 
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

#test_size?
# test_size: 테스트 셋 구성의 비율을 나타냅니다. 
# train_size의 옵션과 반대 관계에 있는 옵션 값이며, 주로 test_size를 지정해 줍니다. 
# 0.2는 전체 데이터 셋의 20%를 test (validation) 셋으로 지정하겠다는 의미입니다. default 값은 0.25 입니다. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train.shape)#(404, 13)
print(x_test.shape)#(102, 13)

x_train=x_train.reshape(404,13, 1, 1)
x_test=x_test.reshape(102,13, 1, 1)


model=Sequential()
model.add(Conv2D(10, (2,2), padding='same' ,input_shape=(13, 1, 1)))
model.add(Conv2D(20, (2,2), padding='same'))
model.add(Conv2D(35, (2,2), padding='same'))
model.add(Conv2D(70, (2,2), padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(50, (2,2), padding='same'))
model.add(Conv2D(30, (2,2), padding='same'))
model.add(Flatten())
model.add(Dense(80, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))

model.summary()

# 회귀 모델은 매트릭스 안주기
model.compile(loss='mse', optimizer='adam')
early_stopping=EarlyStopping(monitor='loss', patience=50, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2 ,callbacks=[early_stopping])

y_predict=model.predict(x_test)

from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score 
r2=r2_score(y_test, y_predict)
print("R2 : ", r2)

'''
RMSE :  3.4807188187633225
R2 :  0.8083720141213773
'''
# import numpy as np
# from sklearn.datasets import load_boston
# dataset = load_boston()

# x = dataset.data
# y = dataset.target

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True) # train size를 70%로 주고 내가 쓴 순서대로 잘려나간다
# # 성능은 셔플을 한게 더 좋다 디폴트는 true   

# x_predict = x_test[:10]
# # y_real = y_test[:10]

# #x_train, x_predict 전처리
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# x_predict = scaler.transform(x_predict)

# print("x_data : ", x)
# print("y_target : ", y)
# print(x.shape) #(506, 13)
# print(y.shape) #(506,)
# print(x_train[0])
# '''
# [  9.82349   0.       18.1       0.        0.671     6.794    98.8
#    1.358    24.      666.       20.2     396.9      21.24   ]
# '''
# print(y_train[0]) #13.3


# # OneHotEncodeing으로 변경된 y_train, y_test 쉐이프 확인하기
# # 대입되어 있는 y_train 값 확인하기
# print("y_train shape : ", y_train.shape)
# print("y_test shape : ", y_test.shape)
# print("y_train data : ", y_train[0])

# # x_train도 확인해 볼거야 y_train[0]도 0을 봤으니 x_train도 [0]을 봐야 하겠지?
# print("x_train data : ", x_train[0])

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Dropout

# model = Sequential()
# model.add(Dense(100, activation='relu', input_shape=(13, ))) #(506, 13)
# model.add(Dense(70, activation='relu'))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(50, activation='relu')) # strides=2 니까 2칸씩 이동한다
# model.add(Dense(30, activation='relu'))
# model.add(Dense(10, activation='relu')) #Dense 디폴트는  linear
# model.add(Dense(1, activation='softmax')) # 꼭 들어감

# model.summary()

# # from tensorflow.keras.callbacks import EarlyStopping
# # early_Stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

# model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2)

# loss = model.evaluate(x_test, y_test, batch_size=1)
# print("loss : ", loss)

# y_predict = model.predict(x_test)
# print("결과물 : \n : ", y_predict)

# # 실습 : 결과물 오차 수정 미세조정
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE : ", RMSE(y_test, y_predict))    

# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print("R2 : : ", r2)

# print(x_test)

# linear = 디폴트 마지막 레이어
# 아웃풋1
# RMSE
# R2



