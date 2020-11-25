from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

diabetes = load_diabetes()

x = diabetes.data
y = diabetes.target

# x = x[:, 3:]
# 1 2 3, 4 5 6 7 8 9 10 

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, test_size=0.2)

print(x_train.shape) #(353, 10)

# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier(max_depth=4)
# model = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier(max_depth=4)

model.fit(x_train, y_train)

y_predict=model.predict(x_test)

print('r2_score : ', r2_score(y_predict, y_test))
print(model.feature_importances_)

'''
import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_diabetes(model) :
    n_features=diabetes.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), diabetes.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_diabetes(model)
plt.show()
'''
'''
r2_score :  -0.12088604796981195
[0.11229191 0.09370205 0.10385654 0.10626341 0.08163328 0.08065958
 0.1225269  0.09262631 0.10440632 0.10203379]
 '''

