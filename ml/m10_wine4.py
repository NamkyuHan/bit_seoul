import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

wine = pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0,)

x = wine.drop('quality', axis=1)
y = wine['quality']

print(x.shape) #(4898, 11)
print(y.shape) #(4898,)

newlist = []
for i in list(y):
    if i <=4: 
        newlist +=[0]
    elif i <=7:
        newlist +=[1]
    else :
        newlist +=[2]

y = newlist

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

#2. 모델
# model = LinearSVC() #분류
# model = SVC() #분류
# model = KNeighborsClassifier() #분류
# model = KNeighborsRegressor() #회귀
model = RandomForestClassifier() #분류
# model = RandomForestRegressor() #회귀

#3. 훈련
model.fit(x_train, y_train)


#4. 평가 예측
score = model.score(x_test, y_test)
# acccuracy_score 를 넣어서 비교할것
print("model.score : ", score)


# 분류일때 쓰자
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

# 회귀모델일 경우 R2_score와 비교할것

# 회귀일때 쓰자
# r2 = r2_score(y_test, y_predict)
# print("R2 : ", r2)

print(y_test[:10], "의 예측결과", '\n', y_predict[:10], "의 실제결과")

'''
RandomForestClassifier()
model.score :  0.9489795918367347
accuracy_score :  0.9489795918367347
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 의 예측결과
[1 1 1 1 1 1 1 1 1 1] 의 실제결과
'''
