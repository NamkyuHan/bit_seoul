from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
print(iris)
print(type(iris)) #<class 'sklearn.utils.Bunch'>

x_data = iris.data
y_data = iris.target

print(type(x_data))
print(type(y_data))

np.save('./data/iris2_x.npy', arr=x_data) # x데이터 저장
np.save('./data/iris2_y.npy', arr=y_data) # y데이터 저장


