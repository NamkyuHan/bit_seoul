# 실습
# 1.상단 모델에 그리드서치 또는 랜덤서치 적용
# 최적의 R2값과 feature_importances 구할것

# 2. 위 쓰레드값으로 SelectFromModel을 구해서 최적의 피처 갯수를 구할 것

# 3. 위 피처 갯수로 데이터(피처)를 수정(삭제)해서 그리드서치 또는 랜덤서치 적용
# 최적의 R2 값을 구할 것

# 1번값과 2번값을 비교해볼것

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
'''
x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)


parameters = [
    {"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.001, 0.01],
    "max_depth":[4, 5, 6]},
    {"n_estimators":[90, 100, 110], "learning_rate":[0.1, 0.001, 0.01],
    "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1]},
    {"n_estimators":[90, 110], "learning_rate":[0.1, 0.001, 0.5],
    "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1],
    "colsample_bylevel":[0.6, 0.7, 0.9]}
]

# 2. 모델
# 그리드 서치는 파라미터를 뽑는거다
model = GridSearchCV(XGBRegressor(), parameters, verbose=2)
# model = XGBRegressor(n_jobs=-1)

# 훈련
model.fit(x_train, y_train)

# 평가 예측
print("최적의 매개변수 : ", model.best_estimator_)
y_predict=model.predict(x_test)
print('최종정답률 : ', r2_score(y_test, y_predict))
print('feature_importances_ : ', XGBRegressor().fit(x_train, y_train).feature_importances_)
'''
'''
최적의 매개변수 뽑기
최적의 매개변수 :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=6,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=110, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
최종정답률 :  0.9311752979545836
feature_importances_ :  [0.01447935 0.00363372 0.01479119 0.00134153 0.06949984 0.30128643
 0.01220458 0.0518254  0.0175432  0.03041655 0.04246345 0.01203115
 0.42848358]
'''
'''
x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=6,
             min_child_weight=1, monotone_constraints='()',
             n_estimators=110, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)

# 훈련
model.fit(x_train, y_train)

# 평가 예측
y_predict=model.predict(x_test)
print('R2 score : ', r2_score(y_test, y_predict))

# 이 sort로 feature_importances를 가려낸다
thresholds = np.sort(model.feature_importances_)
print('feature_importance_ sort : ', thresholds)


for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)
    
    print("thresh=%.3f, n=%d, R2 : %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
    '''
'''
R2 score :  0.9311752979545836

feature_importance_ sort :  [0.00152491 0.00226998 0.00988215 0.01005833 0.0132021  0.01668662
 0.02487051 0.0313093  0.04498584 0.05127558 0.24242924 0.27001473
 0.28149077]
13개
thresh=0.002, n=13, R2 : 92.21%
thresh=0.002, n=12, R2 : 92.16%
thresh=0.010, n=11, R2 : 92.03%
thresh=0.010, n=10, R2 : 92.19%
thresh=0.013, n=9, R2 : 92.59%
thresh=0.017, n=8, R2 : 92.71%
thresh=0.025, n=7, R2 : 92.86%
thresh=0.031, n=6, R2 : 92.71%
thresh=0.045, n=5, R2 : 91.74%
thresh=0.051, n=4, R2 : 91.47%
thresh=0.242, n=3, R2 : 78.35%
thresh=0.270, n=2, R2 : 69.41%
thresh=0.281, n=1, R2 : 44.98%
'''

x, y=load_boston(return_X_y=True)
x_data1=x[:,:4]
x_data2=x[:,5:]
x=np.concatenate([x_data1, x_data2], axis=1)
print(x.shape) #(506, 12)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

parameters= [
    {'n_estimators' : [100,200, 300],
    'learning_rate' : [0.1,0.3,0.001,0.01],
    'max_depth' : [4,5,6]}, 
    {'n_estimators' : [100,200, 300],
    'learning_rate' : [0.1, 0.001, 0.01],
    'max_depth' : [4,5,6],
    'colsample_bytree' :[0.6, 0.9, 1]},
    {'n_estimators' : [90, 110],
    'learning_rate' : [0.1, 0.001, 0.5],
    'max_depth' : [4,5,6],
    'colsample_bytree' :[0.6, 0.9, 1],
    'colsample_bylevel' :[0.6, 0.7, 0.9]}
] 

#2. 모델
model=GridSearchCV(XGBRegressor(), parameters) 


#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
y_predict=model.predict(x_test)
print('최종정답률 : ', r2_score(y_test, y_predict))
print('feature_importance : ', XGBRegressor().fit(x_train, y_train).feature_importances_)

'''
최적의 매개변수 :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=4,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=110, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
최종정답률 :  0.9106800231824272
feature_importance :  [0.01845236 0.00092534 0.00870658 0.00115268 0.29452    0.01097542
 0.05462902 0.01498002 0.03490826 0.04061279 0.0091564  0.51098114]
 '''