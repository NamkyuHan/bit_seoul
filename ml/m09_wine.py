import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


#1. 데이터
x, y = load_wine(return_X_y=True)

'''
datasets = load_wine()
print(datasets.feature_names)
print(datasets.target_names)
'''
'''
['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 
'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
['class_0' 'class_1' 'class_2']
'''


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
model.score :  0.9722222222222222   
accuracy_score :  0.9722222222222222
[2 1 1 0 1 1 2 0 0 1] 의 예측결과
 [2 1 1 0 1 1 2 0 0 0] 의 실제결과

2. SVC
model.score :  1.0
accuracy_score :  1.0
[2 1 1 0 1 1 2 0 0 1] 의 예측결과
 [2 1 1 0 1 1 2 0 0 1] 의 실제결과

3. KNeighborsClassifier
model.score :  1.0   
accuracy_score :  1.0
[2 1 1 0 1 1 2 0 0 1] 의 예측결과
 [2 1 1 0 1 1 2 0 0 1] 의 실제결과

4. KNeighborsRegressor()
model.score :  0.9775954738330976
R2 :  0.9775954738330976
[2 1 1 0 1 1 2 0 0 1] 의 예측결과
 [2.  1.  1.  0.  1.  1.  2.  0.  0.  1.4] 의 실제결과

5. RandomForestClassifier
model.score :  1.0   
accuracy_score :  1.0
[2 1 1 0 1 1 2 0 0 1] 의 예측결과
 [2 1 1 0 1 1 2 0 0 1] 의 실제결과

6. RandomForestRegressor
model.score :  0.9828096181046676
R2 :  0.9828096181046676
[2 1 1 0 1 1 2 0 0 1] 의 예측결과
 [1.87 1.   1.   0.   1.01 1.   1.93 0.   0.   1.07] 의 실제결과

'''
