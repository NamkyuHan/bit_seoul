# 남자 여자 구분하기를
# 넘파이 저장
# fit_generator로 코딩
# 넘파이 불러와서
# .fit으로 코딩

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
xy= train_datagen.flow_from_directory(
    './data/data2', 
    target_size=(150, 150),
    batch_size = 200,
    class_mode = 'binary'
    # save_to_dir='./data/data2/train'
)

print(xy[0][0].shape) # 배치 10으로 했을때 (10, 150, 150, 3)
print(xy[0][1].shape) # 배치 10으로 했을때 (10, ) 얘는 y값이다

np.save('./data/keras64_x.npy', arr=xy[0][0]) # (10, 150, 150, 3)
np.save('./data/keras64_y.npy', arr=xy[0][1]) # (10,)
