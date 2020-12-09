from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

model = XGBRegressor(n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("R2의 값은? : ", score)

# 이 sort로 feature_importances를 가려낸다
thresholds = np.sort(model.feature_importances_)
print(thresholds)

import time
start1 = time.time()

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)

    selection_model = XGBRegressor(n_jobs=-1) #코어 숫자 정하기 쓰레드는 적용 안됨
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)
    
    print("thresh=%.3f, n=%d, R2 : %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

start2 = time.time()
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)

    selection_model = XGBRegressor(n_jobs=6)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("thresh=%.3f, n=%d, R2 : %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

end = start2 - start1
print('그냥 걸린 시간 : ', end)
end2 = time.time() -start2
print('잡스 걸린 시간 : ', end2)

'''
n은 피처의 개수
모델은 n = 피처의 값 만큼 실행시킨 것
thresh=0.001, n=13, R2 : 92.21%
thresh=0.004, n=12, R2 : 92.16%
thresh=0.012, n=11, R2 : 92.03%
thresh=0.012, n=10, R2 : 92.19%
thresh=0.014, n=9, R2 : 93.08%
thresh=0.015, n=8, R2 : 92.37%
thresh=0.018, n=7, R2 : 91.48%
thresh=0.030, n=6, R2 : 92.71%
thresh=0.042, n=5, R2 : 91.74%
thresh=0.052, n=4, R2 : 92.11%
thresh=0.069, n=3, R2 : 92.52%
thresh=0.301, n=2, R2 : 69.41%
thresh=0.428, n=1, R2 : 44.98%
'''



