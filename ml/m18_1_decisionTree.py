from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=66, shuffle=True, test_size=0.2)

print(x_train.shape)

model = DecisionTreeClassifier(max_depth=4)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(acc)
print(model.feature_importances_)

'''
0.9122807017543859
[0.         0.0624678  0.         0.         0.         0.
 0.         0.02364429 0.         0.         0.01297421 0.
 0.01231473 0.         0.         0.         0.         0.
 0.         0.         0.         0.01695087 0.         0.75156772
 0.         0.         0.00485651 0.11522388 0.         0.        ]
 '''

