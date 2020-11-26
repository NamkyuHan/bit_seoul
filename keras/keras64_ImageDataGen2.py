# 넘파이 불러와서
# .fit으로 코딩

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

x=np.load('./data/keras64_x.npy')
y=np.load('./data/keras64_y.npy')

print(x.shape) #(1736, 150, 150, 3)
print(y.shape) #(1736,)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

model=Sequential()
model.add(Conv2D(50, (2,2), input_shape=(150,150,3)))
model.add(Conv2D(100, (3,3)))
model.add(Conv2D(70, (2,2)))
model.add(Conv2D(50, (2,2)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2)

loss, acc=model.evaluate(x_test, y_test, batch_size=100)

print('loss:' ,loss)
print('acc :', acc)

'''
loss: 1.3021094799041748
acc : 0.44999998807907104
'''