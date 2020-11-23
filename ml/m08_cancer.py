import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

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
model.score :  0.9736842105263158   
accuracy_score :  0.9736842105263158
[1 1 1 1 1 0 0 1 1 1] 의 예측결과
 [1 1 1 1 1 0 0 1 1 1] 의 실제결과

2. SVC
model.score :  0.9649122807017544
accuracy_score :  0.9649122807017544
[1 1 1 1 1 0 0 1 1 1] 의 예측결과
 [1 1 1 1 1 0 0 1 1 1] 의 실제결과

3. KNeighborsClassifier
model.score :  0.956140350877193   
accuracy_score :  0.956140350877193
[1 1 1 1 1 0 0 1 1 1] 의 예측결과
 [1 1 1 1 1 0 0 1 1 1] 의 실제결과

4. KNeighborsRegressor()
model.score :  0.8095556298028733
R2 :  0.8095556298028733
[1 1 1 1 1 0 0 1 1 1] 의 예측결과
 [1. 1. 1. 1. 1. 0. 0. 1. 1. 1.] 의 실제결과

5. RandomForestClassifier
model.score :  0.956140350877193   
accuracy_score :  0.956140350877193
[1 1 1 1 1 0 0 1 1 1] 의 예측결과
 [1 1 1 1 1 0 0 1 1 1] 의 실제결과

6. RandomForestRegressor
model.score :  0.8579856331440026
R2 :  0.8579856331440026
[1 1 1 1 1 0 0 1 1 1] 의 예측결과
 [1.   1.   1.   1.   1.   0.16 0.   0.55 0.95 0.99] 의 실제결과

'''
