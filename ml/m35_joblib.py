from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=66, shuffle=True, test_size=0.2)

print(x_train.shape)

# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier(max_depth=4)
# model = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier(max_depth=4)


model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print('acc : ', acc)

# import pickle
# 모델과 가중치 저장 (다른것들도 다 저장 가능 파이썬에서 제공하는 저장모듈임)
# 예시는 bat로 했지만 다른 확장자도 활용가능
# pickle.dump(model, open('./save/xgb_save/cancer.pickle.dat', 'wb'))

import joblib
joblib.dump(model, './save/xgb_save/cancer.joblib.dat')

print('저장됐다')

# model2 = pickle.load(open('./save/xgb_save/cancer.pickle.dat', 'rb'))
model2 = joblib.load('./save/xgb_save/cancer.joblib.dat')
print('불러왔다')

acc2 = model2.score(x_test, y_test)
print('acc2 : ', acc2)
