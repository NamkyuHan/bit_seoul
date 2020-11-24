# 분류
# iris 데이터

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
# from sklearn.utils import all_estimators
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


#1. 데이터

iris = pd.read_csv('./data/csv/iris_ys.csv', header=0, 
                        index_col=0, # 컬럼 번호
                        encoding='CP949',
                        sep=',') # 구분 기호)

x = iris.iloc[:, :4]
y = iris.iloc[:, -1]

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

parameters = [
    {"C": [1,10,100,1000], "kernel":["linear"]},
    {"C": [1,10,100,1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
    {"C": [1,10,100,1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
]

#2. 모델
# 5조각으로 쪼갤거야 셔플은 트루 섞는다
kfold = KFold(n_splits=5, shuffle=True)
# model = SVC()
model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=2)# SVC라는 모델을 파라미터로 쓰겠다

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
print("최적의 매개변수 : ", model.best_estimator_) #베스트 평가자 뽑아줘

y_predict = model.predict(x_test)
print("최종 정답률 : ", accuracy_score(y_test, y_predict))

'''
최적의 매개변수 :  SVC(C=1000, gamma=0.001, kernel='sigmoid')
최종 정답률 :  0.9666666666666667
'''