#1. 데이터
import numpy as np

x1=np.array((range(1,101), range(711, 811), range(100)))
y1=np.array((range(101,201), range(311,411), range(100)))

x1=np.transpose(x1)
y1=np.transpose(y1)

# print(x1.shape) #(100, 3)

x2=np.array((range(1,101), range(761, 861), range(100)))
y2=np.array((range(501,601), range(431,531), range(100,200)))

x2=np.transpose(x2)
y2=np.transpose(y2)

# print(x2.shape) #(100, 3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1,y1, shuffle=True, train_size=0.7
)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2,y2, shuffle=True, train_size=0.7
)

#2. 함수형 모델 2개 구성

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

# 모델1
input1 = Input(shape=(3,))
dense1_1 = Dense(10, activation='relu', name='king1')(input1)
dense1_2 = Dense(7, activation='relu', name='king2')(dense1_1)
dense1_3 = Dense(5, activation='relu', name='king3')(dense1_2)
output1 = Dense(3, activation='linear', name='king4')(dense1_3)

# model1 = Model(inputs=input1, outputs=output1)

# model1.summary()

# 모델2
input2 = Input(shape=(3,))
dense2_1 = Dense(15, activation='relu', name='queen1')(input2)
dense2_2 = Dense(11, activation='relu', name='queen2')(dense2_1)
output2 = Dense(3, activation='linear', name='queen3')(dense2_2) #activation='linear'인 상태

# model2 = Model(inputs=input2, outputs=output2)

# model2.summary()
#----------------------------------------------------------------------------------

#모델 병합, concatenate
from tensorflow.keras.layers import Concatenate, concatenate
# from keras.layers.merge import Concatenate, concatenate
# from keras.layers import Concatenate, concatenate

# merge1 = concatenate([output1, output2]) #2개 이상이라 list로 묶습니다
merge1 = Concatenate(axis=1)([output1, output2])

# middle1 = Dense(30)(merge1)
# middle2 = Dense(7)(middle1)
# middle3 = Dense(11)(middle2)

#이름 이것도 가능 (다만, 가독성 위해 이름을 middle 1, 2, 3)
middle1 = Dense(30, name='middle1')(merge1)
middle2 = Dense(7, name='middle2')(middle1)
middle3 = Dense(11, name='middle3')(middle2)

################# output 모델 구성 (분기)
output1 = Dense(30, name='output1')(middle3)
output1_1 = Dense(7, name='output1_1')(output1)
output1_2 = Dense(3, name='output1_2')(output1_1)

output2 = Dense(15, name='output2')(middle3)
output2_1 = Dense(14, name='output2_1')(output2)
output2_2 = Dense(11, name='output2_2')(output2_1)
output2_3 = Dense(3, name='output2_3')(output2_2)

# 모델 정의
model = Model(inputs = [input1, input2], 
              outputs = [output1, output2_3])

model.summary()

