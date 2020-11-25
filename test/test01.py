print("헬로우 월드")

a=1
b=2

c=a+b
print(c)



import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.append(x_train, x_test, axis=0)

print(x.shape)