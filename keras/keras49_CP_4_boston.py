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
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target
# print(x.shape, y.shape) #(506, 13) (506,)

#test_size?
# test_size: 테스트 셋 구성의 비율을 나타냅니다. 
# train_size의 옵션과 반대 관계에 있는 옵션 값이며, 주로 test_size를 지정해 줍니다. 
# 0.2는 전체 데이터 셋의 20%를 test (validation) 셋으로 지정하겠다는 의미입니다. default 값은 0.25 입니다. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# x데이터 train, test 나눴으니 트레인 데이터 삽입 명시  
# train과 test만 나눴으니 이렇게 해야 함 
scaler = MinMaxScaler()
scaler.fit(x_train)
x = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#데이터 셋이 2차원이기 때문에 reshape 안함
model=Sequential()
model.add(Dense(80, activation='relu', input_shape=(13,)))
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
