# 나머지 데이터셋 6개를 저장하기
# mnist (했음) x_train, x_test, y_train, y_test
# cifar10 x_train, x_test, y_train, y_test
# fashion x_train, x_test, y_train, y_test
# cifar100 x_train, x_test, y_train, y_test
# boston x,y
# diabetes x,y
# iris (했음) x,y
# cancer x,y

from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import cifar100
from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
import numpy as np
'''
# cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

np.save('./data/cifar10_x_train.npy', arr=x_train)
np.save('./data/cifar10_x_test.npy', arr=x_test)
np.save('./data/cifar10_y_train.npy', arr=y_train)
np.save('./data/cifar10_y_test.npy', arr=y_test)

# fashion
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

np.save('./data/fashion_x_train.npy', arr=x_train)
np.save('./data/fashion_x_test.npy', arr=x_test)
np.save('./data/fashion_y_train.npy', arr=y_train)
np.save('./data/fashion_y_test.npy', arr=y_test)

# cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

np.save('./data/cifar100_x_train.npy', arr=x_train)
np.save('./data/cifar100_x_test.npy', arr=x_test)
np.save('./data/cifar100_y_train.npy', arr=y_train)
np.save('./data/cifar100_y_test.npy', arr=y_test)
'''
'''
# boston
boston = load_boston()
print(boston)
print(type(boston)) #<class 'sklearn.utils.Bunch'>

x_data = boston.data
y_data = boston.target

print(type(x_data)) #<class 'numpy.ndarray'>
print(type(y_data)) #<class 'numpy.ndarray'>

np.save('./data/boston_x.npy', arr=x_data) # x데이터 저장
np.save('./data/boston_y.npy', arr=y_data) # y데이터 저장
'''
'''
# diabetes
diabetes = load_diabetes()
print(diabetes)
print(type(diabetes)) #<class 'sklearn.utils.Bunch'>

x_data = diabetes.data
y_data = diabetes.target

print(type(x_data)) #<class 'numpy.ndarray'>
print(type(y_data)) #<class 'numpy.ndarray'>

np.save('./data/diabetes_x.npy', arr=x_data) # x데이터 저장
np.save('./data/diabetes_y.npy', arr=y_data) # y데이터 저장
'''
'''
# iris
iris = load_iris()
print(iris)
print(type(iris)) #<class 'sklearn.utils.Bunch'>

x_data = iris.data
y_data = iris.target

print(type(x_data)) #<class 'numpy.ndarray'>
print(type(y_data)) #<class 'numpy.ndarray'>

np.save('./data/iris_x.npy', arr=x_data) # x데이터 저장
np.save('./data/iris_y.npy', arr=y_data) # y데이터 저장
'''
'''
# cancer
cancer = load_breast_cancer()
print(cancer)
print(type(cancer)) #<class 'sklearn.utils.Bunch'>

x_data = cancer.data
y_data = cancer.target

print(type(x_data)) #<class 'numpy.ndarray'>
print(type(y_data)) #<class 'numpy.ndarray'>

np.save('./data/cancer_x.npy', arr=x_data) # x데이터 저장
np.save('./data/cancer_y.npy', arr=y_data) # y데이터 저장
'''










