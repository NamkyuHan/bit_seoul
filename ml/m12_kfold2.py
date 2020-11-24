# 4개의 모델을 완성
#2. 모델
# model = LinearSVC() #분류
# model = SVC() #분류
# model = KNeighborsClassifier() #분류
# model = KNeighborsRegressor() #회귀
# model = RandomForestClassifier() #분류
# model = RandomForestRegressor() #회귀

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

#1. 데이터


iris = pd.read_csv('./data/csv/iris_ys.csv', header=0)

x = iris.iloc[:, :4]
y = iris.iloc[:, -1]

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.2)


from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#2. 모델
# 5조각으로 쪼갤거야 셔플은 트루 섞는다
kfold = KFold(n_splits=5, shuffle=True)

model = SVC()

scores = cross_val_score(model, x_train, y_train, cv=kfold)

print('SVC()_score : ', scores)


kfold = KFold(n_splits=5, shuffle=True)

#=====================================================================

model = LinearSVC()

scores = cross_val_score(model, x_train, y_train, cv=kfold)

print('LinearSVC()_score : ', scores)

#=====================================================================

model = KNeighborsClassifier()

scores = cross_val_score(model, x_train, y_train, cv=kfold)

print('KNeighborsClassifier()_score : ', scores)

#=====================================================================

model = KNeighborsRegressor()

scores = cross_val_score(model, x_train, y_train, cv=kfold)

print('KNeighborsRegressor()_score : ', scores)

#=====================================================================

model = RandomForestClassifier()

scores = cross_val_score(model, x_train, y_train, cv=kfold)

print('RandomForestClassifier()_score : ', scores)

#=====================================================================

model = RandomForestRegressor()

scores = cross_val_score(model, x_train, y_train, cv=kfold)

print('RandomForestRegressor()_score : ', scores)



'''
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


