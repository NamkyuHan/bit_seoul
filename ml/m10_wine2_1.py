#RF로 모델을 만들것

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

wine = pd.read_csv('./data/csv/winequality-white.csv', 
                        header=0, # 첫 번 째 행 = 헤더다
                        index_col=0, # 컬럼 번호
                        sep=';',  # 구분 기호
                        encoding='cp949')
# print(df1)
# print(df1.shape)
# print(type(df1))
# print(df1.describe())

#[4898 rows x 1 columns]
#(4898, 1)
# <class 'pandas.core.frame.DataFrame'>

# 판다스 넘파이로 바꿈
# 마지막 값 하나씩 빼서 볼거야
wine = wine.to_numpy()
x = wine[:,:-1]
y = wine[:,-1]


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print("x_train.shape",x_train.shape)
print("x_test.shape",x_test.shape)



#2. 모델
# model = LinearSVC() #분류
# model = SVC() #분류
# model = KNeighborsClassifier() #분류
# model = KNeighborsRegressor() #회귀
model = RandomForestClassifier() #분류
# model = RandomForestRegressor() #회귀

#3. 훈련
model.fit(x_train, y_train)


#4. 평가 예측
score = model.score(x_test, y_test)
# acccuracy_score 를 넣어서 비교할것
print("model.score : ", score)


# 분류일때 쓰자
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

# 회귀모델일 경우 R2_score와 비교할것

# 회귀일때 쓰자
# r2 = r2_score(y_test, y_predict)
# print("R2 : ", r2)

print(y_test[:10], "의 예측결과", '\n', y_predict[:10], "의 실제결과")

'''
RandomForestClassifier
model.score :  0.7142857142857143
accuracy_score :  0.7142857142857143
[6. 6. 6. 6. 6. 5. 5. 7. 5. 6.] 의 예측결과
[6. 6. 6. 6. 6. 5. 5. 7. 5. 6.] 의 실제결과

RandomForestRegressor()
model.score :  0.5539164328567285
R2 :  0.5539164328567285
[6. 6. 6. 6. 6. 5. 5. 7. 5. 6.] 의 예측결과
 [6.24 6.15 6.02 6.14 5.79 5.43 5.47 6.86 5.17 5.57] 의 실제결과
'''






