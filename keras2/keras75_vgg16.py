from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Sequential

# model = VGG16() # 파라미터 개수 138, 537, 544
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
# trainable=False 면 더 훈련 시키지 않을거야 이미지넷 가중치 사용할거야 true면 반대
vgg16.trainable=False

# vgg16.summary()

# 레이어당 웨이트 바이어스 보여준다
# print("동결하기 전 훈련되는 가중치의 수 ", len(vgg16.trainable_weights)) #32

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256))
# model.add(BatchNormalization)
# model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Dense(10, activation='softmax'))

model.summary()

print("동결하기 전 훈련되는 가중치의 수 ", len(model.trainable_weights)) 
# print(model.trainable_weights)

# 각 레이어의 정보를 볼 수 있는 코드
import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

# print(aaa.loc[:][2:])
print(aaa)

