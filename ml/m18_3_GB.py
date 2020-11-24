from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=66, shuffle=True, test_size=0.2)

print(x_train.shape)

# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier(max_depth=4)
model = GradientBoostingClassifier(max_depth=4)


model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(acc)
print(model.feature_importances_)

'''
0.9649122807017544
[3.63382521e-04 6.25911141e-02 3.35587675e-03 1.56679049e-04
 3.61883787e-03 1.41091225e-04 9.95646114e-04 7.79709104e-02
 1.36396506e-04 4.79362825e-04 5.86216820e-03 8.45536805e-05
 2.76337919e-03 1.17307687e-02 3.02601504e-03 4.77087578e-04
 6.41565330e-03 9.51464047e-04 4.39021802e-05 2.67930454e-03
 4.48916528e-02 2.55926668e-02 2.17788089e-03 6.14985886e-01
 2.42206729e-03 9.90117299e-05 8.42719657e-03 1.12959966e-01
 1.12744363e-03 3.47263532e-03]
 '''

