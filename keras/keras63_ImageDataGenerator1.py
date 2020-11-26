from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 생성 옵션 정하기
train_datagen = ImageDataGenerator(rescale=1./255, 
                                horizontal_flip=True,
                                vertical_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                rotation_range=5,
                                zoom_range=1.2,
                                shear_range=0.7,
                                fill_mode='nearest'
                                )
test_datagen = ImageDataGenerator(rescale=1./255)

# flow 또는 flow_from_directory
# 실제 데이터가 있는 곳을 알려주고, 이미지를 불러오는 작업
xy_train=train_datagen.flow_from_directory(
    './data/data1/train', #실제 이미지가 있는 폴더는 라벨이 됨. (ad/normal=0/1)
    target_size=(150,150),
    batch_size=5,
    class_mode='binary' 
) # x와 y가 이미 갖춰진 데이터셋

xy_test=test_datagen.flow_from_directory(
    './data/data1/test',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary' 
)



# train 셋에 이미 x와 y가 존재하므로 하나만 써주면 됨
# model.fit_generator(
#     xy_train,
#     steps_per_epoch=100,
#     epochs=20,
#     validation_data=xy_test, validation_steps=4
# )