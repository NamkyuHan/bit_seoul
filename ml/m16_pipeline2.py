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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.utils import all_estimators


#1. 데이터

iris = pd.read_csv('./data/csv/iris_ys.csv', header=0, 
                        index_col=0, # 컬럼 번호
                        encoding='CP949',
                        sep=',') # 구분 기호)

x = iris.iloc[:, :4]
y = iris.iloc[:, -1]

# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)


parameters = [
    {"malDDong__C": [1,10,100,1000], "malDDong__kernel":["linear"]},
    {"malDDong__C": [1,10,100,1000], "malDDong__kernel":["rbf"], "malDDong__gamma":[0.001, 0.0001]},
    {"malDDong__C": [1,10,100,1000], "malDDong__kernel":["sigmoid"], "malDDong__gamma":[0.001, 0.0001]}
]


#2. 모델
# pipe = make_pipeline(MinMaxScaler(), SVC())
pipe = Pipeline([("scaler", MinMaxScaler()), ('malDDong', SVC())])

model = RandomizedSearchCV(pipe, parameters, cv=5) 
#3. 훈련
model.fit(x_train, y_train)

#4. 평가
print('acc : ', model.score(x_test, y_test))

print("최적의 매개변수 : ", model.best_estimator_) #베스트 평가자 뽑아줘


