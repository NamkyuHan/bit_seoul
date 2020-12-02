import numpy as np
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout

####1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_predict=x_test[:10, :, :, :]

x_train = x_train.reshape(60000,28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000,28, 28, 1).astype('float32')/255.
# x_predict=x_predict.astype('float32')/255.

# 원 핫 인코딩
y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)


#2. 모델 구성
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28,28,1), name='input')
    x = Conv2D(30, (2,2), padding='same', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(20, (3,3), padding='valid', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Conv2D(10, (2,2), name='hidden3')(x)
    x = Dropout(drop)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')

    return model

# 이 함수 부분이 중요하다 파라미터를 지정해주는 함수니까
# 노드의 갯수와 파라미터 값을 설정할 수 있다 그러므로 위의 모델에서 레이어를 적절히 구성해야 한다
def create_hyperparameters():
    batchs = [10] #[10, 20, 30, 40, 50] # [10]
    optimizers = ('rmsprop', 'adam', 'adadelta') #['rmsprop']
    # dropout = np.linspace(0.1, 0.5, 5)
    dropout = [0.1, 0.5, 5]
    return{'batch_size' : batchs, "optimizer" : optimizers, "drop" : dropout}
hyperparamaters = create_hyperparameters()


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier #케라스를 사이킷런으로 감싸겠다
model = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparamaters, cv=3)    
search.fit(x_train, y_train)

print(search.best_params_)
acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc)

'''
{'batch_size': 40, 'drop': 0.1, 'optimizer': 'adam'}
250/250 [==============================] - 0s 989us/step - loss: 0.1147 - acc: 0.9635
최종 스코어 :  0.9635000228881836
'''
# 그리드 서치를 랜덤 서치로 바꿔보자

