from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=66, shuffle=True, test_size=0.2)

print(x_train.shape)

# model = DecisionTreeClassifier(max_depth=4)
model = RandomForestClassifier(max_depth=4)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(acc)
print(model.feature_importances_)

'''
0.956140350877193
[0.04088637 0.0126763  0.04059879 0.04108537 0.00305843 0.01296661
 0.06508106 0.10926484 0.00214317 0.00274653 0.02105235 0.00464659
 0.00466563 0.0438295  0.00250974 0.00291056 0.0054002  0.00195336
 0.00409265 0.00289216 0.09029653 0.01820046 0.19821504 0.10083375
 0.00770278 0.00952591 0.04278193 0.0951976  0.00784456 0.00494124]
 '''

