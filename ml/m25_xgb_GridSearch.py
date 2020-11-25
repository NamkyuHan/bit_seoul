# 과적합 방지
# 1. 훈련데이터량을 늘린다
# 2. 피처수를 줄인다
# 3.

from sklearn.datasets import load_boston
from xgboost import XGBClassifier, XGBRegressor, plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score


#회귀 모델
boston = load_boston()

x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, test_size=0.2)

parameters = [
    {"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.001, 0.01],
    "max_depth":[4, 5, 6]},
    {"n_estimators":[90, 100, 110], "learning_rate":[0.1, 0.001, 0.01],
    "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1]},
    {"n_estimators":[90, 110], "learning_rate":[0.1, 0.001, 0.5],
    "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1],
    "colsample_bylevel":[0.6, 0.7, 0.9]}
]

#2. 모델
model = GridSearchCV(XGBRegressor(), parameters, verbose=2)# RandomForestClassifier라는 모델을 파라미터로 쓰겠다
# model = XGBRegressor(max_depth=4, parameters, verbose=2)


model.fit(x_train, y_train)

# score = model.score(x_test, y_test)
# print("점수 : ", score)
# print(model.feature_importances_)

# y_predict = model.predict(x_test)
# print("최종 정답률 : ", r2_score(y_test, y_predict))

print("최적의 매개변수 : ", model.best_estimator_)
y_predict=model.predict(x_test)
print('최종정답률 : ', r2_score(y_test, y_predict))
'''
최적의 매개변수 :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=6,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=110, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
최종정답률 :  0.9311752979545836
'''