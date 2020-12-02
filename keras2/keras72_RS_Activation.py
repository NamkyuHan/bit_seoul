# lr 넣고
import numpy as np
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
# 액티베이션 임포트는 이렇게
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU, ReLU, ELU
from tensorflow.keras.activations import relu, selu, elu

####1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_predict=x_test[:10, :, :, :]

x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test = x_test.reshape(10000,28*28).astype('float32')/255.
# x_predict=x_predict.astype('float32')/255.

# 원 핫 인코딩
y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)


#2. 모델 구성
def build_model(drop=0.5, optimizer=Adam, learning_num=0.001):
    inputs = Input(shape=(28*28, ), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, name='hidden2')(x)
    x = Activation('elu')(x)
    x = Dropout(drop)(x)
    x = Dense(128, name='hidden3')(x)
    x = LeakyReLU(alpha=0.3)(x)
    outputs = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizer(lr=learning_num), metrics=['acc'], loss='categorical_crossentropy')

    return model

# 이 함수 부분이 중요하다 파라미터를 지정해주는 함수니까
# 노드의 갯수와 파라미터 값을 설정할 수 있다 그러므로 위의 모델에서 레이어를 적절히 구성해야 한다
def create_hyperparameters():
    batchs = [10] #[10, 20, 30, 40, 50] # [10]
    optimizers = (Adam, RMSprop) #['rmsprop']
    # dropout = np.linspace(0.1, 0.5, 5)
    dropout = [0.2] #[0.1, 0.5, 5]
    epochs = [10]
    learning = [0.001, 0.01, 0.1]
    return_parameter = {'batch_size' : batchs, "optimizer" : optimizers, "drop" : dropout, "learning_num" : learning, "epochs" : epochs}
    return return_parameter
hyperparamaters = create_hyperparameters()


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier #케라스를 사이킷런으로 감싸겠다
model = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = GridSearchCV(model, hyperparamaters, cv=3)    
search.fit(x_train, y_train)

print(search.best_params_)
acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc)

# 그리드 서치를 랜덤 서치로 바꿔보자

'''
{'batch_size': 10, 'drop': 0.2, 'epochs': 10, 'learning_num': 0.001, 'optimizer': <class 'tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop'>}
   1/1000 [..............................] - ETA: 0s - loss: 2.3063e-04 - acc: 1.0000WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0003s vs `on_test_batch_end` time: 0.0010s). Check your callbacks.
1000/1000 [==============================] - 1s 949us/step - loss: 0.1257 - acc: 0.9814
최종 스코어 :  0.9814000129699707
'''





















'''
################### 1. load_model ########################

#3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_cifar10.h5')
#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)



############## 2. load_model ModelCheckPoint #############

from tensorflow.keras.models import load_model

model2 = load_model('./model/cifar10-04-1.1035.hdf5')
#4. 평가, 예측
result2 = model2.evaluate(x_test, y_test, batch_size=32)


################ 3. load_weights ##################

# 2. 모델
model3 = Sequential()
model3.add(Conv2D(3, (2,2), input_shape=(32,32,3)))
model3.add(Conv2D(20, (2,2)))
model3.add(Conv2D(30, (2,2)))
model3.add(Conv2D(50, (2,2)))
model3.add(Conv2D(70, (2,2)))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Flatten())
model3.add(Dense(20, activation='relu'))
model3.add(Dense(10, activation='softmax')) 

# model.summary()

# 3. 컴파일

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/weight_cifar10.h5')

#4. 평가, 예측

result3 = model3.evaluate(x_test, y_test, batch_size=32)


print("모델 저장 loss : ", result1[0])
print("모델 저장 accuracy : ", result1[1])

print("가중치 저장 loss : ", result3[0])
print("가중치 저장 accuracy : ", result3[1])

print("체크포인트 loss : ", result2[0])
print("체크포인트 accuracy : ", result2[1])


'''