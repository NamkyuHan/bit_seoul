import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


wine = pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0,)

x = wine.drop('quality', axis=1)
y = wine['quality']

# print(x.shape) #(4898, 11)
# print(y.shape) #(4898,)

# 4보다 작거나 같을때 0그룹 7보다 같거나 작을때 1그룹 나머지 2그룹
newlist = []
for i in list(y):
    if i <=4: 
        newlist +=[0]
    elif i <=7:
        newlist +=[1]
    else :
        newlist +=[2]

y = newlist

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

# 스탠다드 스케일러 적용
scale=StandardScaler()
scale.fit(x_train)
x_train=scale.transform(x_train)
x_test=scale.transform(x_test)

# to_categorical 적용
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#2. 모델
# model = LinearSVC() #분류
# model = SVC() #분류
# model = KNeighborsClassifier() #분류
# model = KNeighborsRegressor() #회귀
# model = RandomForestClassifier() #분류
# model = RandomForestRegressor() #회귀

# 그룹을 3개로 나눴으니 아웃풋 3개 단일분류가 아니니까 소프트맥스 적용
model = Sequential()
model.add(Dense(20, input_shape=(11, )))
model.add(Dense(20, activation='relu'))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(22))
model.add(Dense(3, activation='softmax'))



#3. 훈련 컴파일
# 'binary_crossentropy' 이건 단일분류
# 다중분류 일때는 categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=1, epochs=100)


#4. 평가 예측
# score = model.score(x_test, y_test)
# acccuracy_score 를 넣어서 비교할것
# print("model.score : ", score)

loss, acc = model.evaluate(x_test, y_test, batch_size=1)


# 분류일때 쓰자
y_predict = model.predict(x_test)
y_predict=np.argmax(y_predict, axis=1)
y_actually=np.argmax(y_test, axis=1)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)
# print(x_test, '의 예측 결과', y_predict)
print('loss :', loss)
print('acc :', acc)

# 회귀모델일 경우 R2_score와 비교할것

# 회귀일때 쓰자
# r2 = r2_score(y_test, y_predict)
# print("R2 : ", r2)

print(y_actually[:10], "의 실제결과", '\n', y_predict[:10], "의 예측결과")

'''
loss : 0.2930354177951813
acc : 0.9234693646430969
[1 1 1 1 1 1 1 1 1 1] 의 실제결과
 [1 2 1 1 1 1 1 1 1 1] 의 예측결과
 '''
