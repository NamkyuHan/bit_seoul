from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

np.random.seed(33)


# 이미지에 대한 생성 옵션 정하기
# 기본적으로 이미지 증폭이 있다
train_datagen = ImageDataGenerator(rescale=1./255, # 사이즈 줄이기 정규화
                                   horizontal_flip=True, # 수평 조정
                                   vertical_flip=True, # 수직 조정
                                   width_shift_range=0.1, # 가로 길이 조정
                                   height_shift_range=0.1, # 세로 길이 조정
                                   rotation_range=5, # 얼마나 돌릴거니
                                   zoom_range=1.2, # 줌 범위 설정
                                   shear_range=0.7,
                                   fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# flow 또는 flow_from_directory
# 그냥 데이터는 flow로 폴더로부터 가져오는 건 flow_from_directory
# 실제 데이터가 있는 곳을 알려주고, 이미지를 불러오는 작업
xy_train = train_datagen.flow_from_directory(
    './data/data1/train', 
    target_size=(150, 150),
    batch_size = 1,
    class_mode = 'binary'
    # save_to_dir='./data/data2/train'
)

xy_test = test_datagen.flow_from_directory(
    './data/data1/test', 
    target_size=(150, 150),
    batch_size = 1,
    class_mode = 'binary'
)

print("=================================================")
# print(type(xy_train))
# print(xy_train[0])
# # print(xy_train[0].shape) 에러남
# print(xy_train[0][0])
# print(type(xy_train[0][0])) # 타입이 넘파이가 나옴 <class 'numpy.ndarray'>
# print(xy_train[0][0].shape) # (5, 150, 150, 3)
# print(xy_train[0][1].shape) # (5, ) 얘는 y값이다

# print(xy_train[1][0].shape) # (5, 150, 150, 3)
# print(xy_train[1][1].shape) # (5, ) 얘는 y값이다
# 앞에 괄호는 결국 배치사이즈이다

# 전체 렝스 보기
print(len(xy_train))

# x값 첫번째 보기
# print('첫번째 값 : ', xy_train[0][0][0])
# print('첫번째 쉐이프 : ', xy_train[0][0][0].shape) #(150, 150, 3)

# 설정된 배치사이즈로 y값 보기
# print('첫번째 값 : ', xy_train[0][0][0])
# print('첫번째 y값 쉐이프 : ', xy_train[0][1][:5].shape) #(150, 150, 3)

# np.save('./data/keras63_train_x.npy', arr=xy_train[0][0])
# np.save('./data/keras63_train_y.npy', arr=xy_train[0][1])
# np.save('./data/keras63_test_x.npy', arr=xy_test[0][0])
# np.save('./data/keras63_test_y.npy', arr=xy_test[0][1])

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(150,150,3)))
model.add(Conv2D(20, (2,2)))
model.add(Conv2D(30, (3,3)))
model.add(Conv2D(40, (2,2)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit_generator(
    xy_train,
    steps_per_epoch=100, #한 epoch에 사용한 스텝 수를 지정합니다. 배치사이즈랑 비슷한 개념
    epochs=20,
    validation_data = xy_test, validation_steps=4
)

#4. 평가, 예측
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

print(loss)
# print(accuracy)

# 그래프
plt.plot(acc)
plt.plot(val_acc)
plt.plot(loss)
plt.plot(val_loss)

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')

plt.legend(['loss', 'val_loss', 'acc', 'val_acc'])
plt.show()





'''
모델 인풋 150,150,3
아웃풋 댄스 1 시그모이드
시각화도 해야함
'''