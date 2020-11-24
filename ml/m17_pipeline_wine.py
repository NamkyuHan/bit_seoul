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

iris = pd.read_csv('./data/csv/winequality-white.csv', header=0, 
                        index_col=0, # 컬럼 번호
                        encoding='CP949',
                        sep=';') # 구분 기호)

x = iris.iloc[:, :4]
y = iris.iloc[:, -1]

# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)


parameters = [
    {"malDDong__n_estimators": [100,200],
    "malDDong__max_depth": [6,8,10,12],
    "malDDong__min_samples_leaf": [3,5,7,10],
    "malDDong__min_samples_split": [2,3,5,10],
    "malDDong__n_jobs": [-1]}
]


#2. 모델
# pipe = make_pipeline(MinMaxScaler(), SVC())
pipe = Pipeline([("scaler", MinMaxScaler()), ('malDDong', RandomForestClassifier())])

model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=2) 
#3. 훈련
model.fit(x_train, y_train)

#4. 평가
print("최적의 매개변수 : ", model.best_estimator_) #베스트 평가자 뽑아줘

y_predict = model.predict(x_test)
print("최종 정답률 : ", accuracy_score(y_test, y_predict))

'''
최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()),
                ('malDDong',
                 RandomForestClassifier(max_depth=12, min_samples_leaf=3,
                                        min_samples_split=3, n_jobs=-1))])
최종 정답률 :  0.6183673469387755
'''
