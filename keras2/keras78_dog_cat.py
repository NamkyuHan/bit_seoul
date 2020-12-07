from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

img_dog = load_img('./data/dog_cat/개.jpg', target_size=(224, 224))
img_cat = load_img('./data/dog_cat/고양이.jpg', target_size=(224, 224))
img_suit = load_img('./data/dog_cat/슈트.jpg', target_size=(224, 224))
img_pika = load_img('./data/dog_cat/피카츄.png', target_size=(224, 224))
img_jimin = load_img('./data/dog_cat/한지민.png', target_size=(224, 224))

# plt.imshow(img_dog)
# plt.imshow(img_cat)
# plt.show()

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_suit = img_to_array(img_suit)
arr_pika = img_to_array(img_pika)
arr_jimin = img_to_array(img_jimin)

print(arr_dog)
print(type(arr_dog)) #<class 'numpy.ndarray'>
print(arr_dog.shape) #(659, 450, 3)

print(arr_cat)
print(type(arr_cat)) #<class 'numpy.ndarray'>
print(arr_cat.shape) #(659, 450, 3)

# VGG16 모델에 넣으려면 RGB 형태의 이미지를 BGR로 바꿔야 한다
from tensorflow.keras.applications.vgg16 import preprocess_input
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_suit = preprocess_input(arr_suit)
arr_pika = preprocess_input(arr_pika)
arr_jimin = preprocess_input(arr_jimin)

print(arr_dog.shape)
print(arr_cat.shape)

arr_input = np.stack([arr_dog, arr_cat, arr_pika, arr_jimin, arr_suit])
print(arr_input.shape) # (2, 224, 224, 3)

#2. 모델구성
model = VGG16()
prods = model.predict(arr_input)

print(prods)
print('prods.shape : ', prods.shape) # (2, 1000)

# 이미지 결과 확인
from keras.applications.vgg16 import decode_predictions

result = decode_predictions(prods)
print('=============================================')
print('result[0] : ', result[0])
print('=============================================')
print('result[1] : ', result[1])
print('=============================================')
print('result[2] : ', result[2])
print('=============================================')
print('result[3] : ', result[3])
print('=============================================')
print('result[3] : ', result[4])
print('=============================================')
