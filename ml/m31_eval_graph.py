from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

# dataset = load_boston()
# x = dataset.data
# y = dataset.target()

x, y = load_boston(return_X_y=True)

x_train, x_text, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=77)

#2. 모델 구성
# n_estimators 디폴트는 100번
model = XGBRegressor(n_estimators=1000, learning_rate=0.2)
# model = XGBRegressor(learning_rate=0.1)


#3. 훈련
# verbodse=1 진행 과정 다 보여줄게, True도 가능
# 케라스처럼 evaluation 가능, eval_metric='rmse', eval_set=[(x_text, y_test)])
# 매트릭스를 세팅할 수 있고 test데이터로 세팅도 할 수 있다
# 매트릭스는 평가라 훈련에 반영되지 않는다
# 매트릭스 평가지표 종류 : rmse, mae, logloss, error, auc
# 회귀 일때는 rmse 평균 제곱근 오차 (보스턴은 회귀분석)
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse','logloss'], eval_set=[(x_train, y_train), (x_text, y_test)],
        #  early_stopping_rounds=20       
)


#4. 평가 예측
result = model.evals_result()
print("evals_result : ", result)

y_predict = model.predict(x_text)

r2 = r2_score(y_predict, y_test)
print("r2 : ", r2)

# r2 :  0.9131855680070514

# 시각화 부분
import matplotlib.pyplot as plt

epochs = len(result['validation_0']['logloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['logloss'], label='Train')
ax.plot(x_axis, result['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
# plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['logloss'], label='Train')
ax.plot(x_axis, result['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Rmse')
plt.title('XGBoost Rmse')
plt.show()



