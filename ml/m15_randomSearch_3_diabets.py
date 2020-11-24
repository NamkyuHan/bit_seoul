# 당뇨병 데이터 
# 모델은? RandomForestRegressor
# 보스턴은 리그레서

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
# from sklearn.utils import all_estimators
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_diabetes



#1. 데이터

# iris = pd.read_csv('./data/csv/iris_ys.csv', header=0, 
#                         index_col=0, # 컬럼 번호
#                         encoding='CP949',
#                         sep=',') # 구분 기호)

# x = iris.iloc[:, :4]
# y = iris.iloc[:, -1]

# print(x.shape)
# print(y.shape)

diabetes = load_diabetes()

x = diabetes.data
y = diabetes.target
# print(x)
# print(x.shape, y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, test_size=0.2)

# 첫번째 파라미터 이건 다 더하기 15개
parameters = [
    {"n_estimators": [100,200]},
    {"max_depth": [6,8,10,12]},
    {"min_samples_leaf": [3,5,7,10]},
    {"min_samples_split": [2,3,5,10]},
    {"n_jobs": [-1]}
]

# 두번째 파라미터 이건 다 곱하기 128개
# parameters = [
#     {"n_estimators": [100,200],
#     "max_depth": [6,8,10,12],
#     "min_samples_leaf": [3,5,7,10],
#     "min_samples_split": [2,3,5,10],
#     "n_jobs": [-1]}
# ]


#2. 모델
# 5조각으로 쪼갤거야 셔플은 트루 섞는다
kfold = KFold(n_splits=5, shuffle=True)
# model = SVC()
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=2)# RandomForestClassifier라는 모델을 파라미터로 쓰겠다

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
print("최적의 매개변수 : ", model.best_estimator_) #베스트 평가자 뽑아줘

y_predict = model.predict(x_test)
print("최종 정답률 : ", r2_score(y_test, y_predict))


'''
최적의 매개변수 :  RandomForestRegressor()
최종 정답률 :  0.38251188411209924
'''
