import numpy as np
from sklearn.svm import LinearSVC ,SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 1, 1, 0]

#2. 모델
# model = LinearSVC()
# model = SVC()
model = Sequential()
model.add(Dense(20, input_dim=2))
model.add(Dense(20, activation='relu'))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(22))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. 평가 예측
y_predict = model.predict(x_data)
print(x_data, '의 예측 결과', y_predict)

# 이건 실행값
acc1 = model.evaluate(x_data, y_data)
print('model.evaluate : ', acc1)

# 이건 예측값
# acc2 = accuracy_score(y_data, y_predict)
# print('accuracy_score : ', acc2)



