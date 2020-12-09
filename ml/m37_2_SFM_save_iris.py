from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score
import pickle

x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

model = XGBRegressor(n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("R2의 값은? : ", score)

# 이 sort로 feature_importances를 가려낸다
thresholds = np.sort(model.feature_importances_)
print(thresholds)

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)
    
    print("thresh=%.3f, n=%d, R2 : %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
    pickle.dump(model, open('./save/xgb_save/iris2.pickle.dat', 'wb'))
    print('저장됐다')


# 모델과 가중치 저장 (다른것들도 다 저장 가능 파이썬에서 제공하는 저장모듈임)
# 예시는 bat로 했지만 다른 확장자도 활용가능
# pickle.dump(model, open('./save/xgb_save/cancer2.pickle.dat', 'wb'))
# print('저장됐다')

model2 = pickle.load(open('./save/xgb_save/iris2.pickle.dat', 'rb'))
print('불러왔다')
acc2 = model2.score(x_test, y_test)
print('acc2 : ', acc2)


