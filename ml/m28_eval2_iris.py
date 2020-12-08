from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

# dataset = load_boston()
# x = dataset.data
# y = dataset.target()

x, y = load_iris(return_X_y=True)

x_train, x_text, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=77)

#2. 모델 구성
# n_estimators 디폴트는 100번
model = XGBClassifier(n_estimators=500, learning_rate=0.1)
# model = XGBRegressor(learning_rate=0.1)


#3. 훈련
# verbodse=1 진행 과정 다 보여줄게, True도 가능
# 케라스처럼 evaluation 가능, eval_metric='rmse', eval_set=[(x_text, y_test)])
# 매트릭스를 세팅할 수 있고 test데이터로 세팅도 할 수 있다
# 매트릭스는 평가라 훈련에 반영되지 않는다
# 매트릭스 평가지표 종류 : rmse, mae, logloss, error, auc
# 다중분류일때는 merror : 다중 클래스 분류 오류율 (아이리스는 다중분류)
model.fit(x_train, y_train, verbose=1, eval_metric='merror', eval_set=[(x_train, y_train), (x_text, y_test)])


#4. 평가 예측
result = model.evals_result()
print("evals_result : ", result)

y_predict = model.predict(x_text)

acc = accuracy_score(y_predict, y_test)
print("acc : ", acc)

# acc :  0.8333333333333334
