
#모델 저장하기

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#2. 모델
model = Sequential()
model.add(LSTM(100, input_shape=(4,1)))
model.add(Dense(50, name='queen1'))
model.add(Dense(10, name='queen2'))
# model.add(Dense(1 ,name='queen3'))

model.summary()

model.save("./save/keras28.h5")
# model.save(".\save\keras28_2.h5")
# model.save(".//save//keras28_3.h5")
# model.save(".\\save\\keras28_4.h5")

# 이 경로 모두 잘 저장됨
# \n 이거 조심하자