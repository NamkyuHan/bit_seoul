# 과적합 방지
# 1. 훈련데이터량을 늘린다
# 2. 피처수를 줄인다
# 3.

from sklearn.datasets import load_boston
from xgboost import XGBClassifier, XGBRegressor, plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#회귀 모델
boston = load_boston()

x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, test_size=0.2)


n_estimators = 300
learning_rate = 1
colsample_bytree = 1
colsample_bylevel = 1

max_depth = 5
n_jobs = -1

model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate,
                    n_estimators=n_estimators, n_jobs=n_jobs,
                    colsample_bylevel=colsample_bylevel,
                    colsample_bytree=colsample_bytree)

# score 디폴트로 했던놈과 성능 비교.

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("점수 : ", score)
print(model.feature_importances_)

plot_importance(model)
plt.show()

