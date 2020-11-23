import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
model = LinearSVC() #분류
# model = SVC() #분류
# model = KNeighborsClassifier() #분류
# model = KNeighborsRegressor() #회귀
# model = RandomForestClassifier() #분류
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
*모델별 결과에 대해 명시*

1. LinearSVC
model.score :  0.011235955056179775   
accuracy_score :  0.011235955056179775
[235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측결과
 [ 91. 232. 281. 134.  97.  69.  53. 109. 230. 101.] 의 실제결과

2. SVC
model.score :  0.0
accuracy_score :  0.0
[235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측결과
 [ 91. 220.  91.  90.  90.  91.  53.  90. 220. 200.] 의 실제결과

3. KNeighborsClassifier
model.score :  0.0   
accuracy_score :  0.0
[235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측결과
 [ 91. 129.  77.  64.  60.  63.  42.  53.  67.  74.] 의 실제결과

4. KNeighborsRegressor()
model.score :  0.38626977834604637
R2 :  0.38626977834604637
[235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측결과
 [166.4 190.6 124.  132.6 120.4 120.2 101.4 108.6 145.4 131.6] 의 실제결과

5. RandomForestClassifier
model.score :  0.011235955056179775   
accuracy_score :  0.011235955056179775
[235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측결과
 [ 91. 163.  77.  95.  53.  91.  71. 214.  67. 101.] 의 실제결과

6. RandomForestRegressor
model.score :  0.3658186954908037
R2 :  0.3658186954908037
[235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측결과
 [156.11 207.95 163.3  136.31  97.49 122.41  98.26 137.46 140.37 116.4 ] 의 실제결과

'''
