import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Conv1D, MaxPooling1D, Flatten, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn. preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras62_split2 import split_data

#삼성전자
df1 = pd.read_csv('./data/csv/삼성전자 1120.csv', header=0, index_col=None, sep=',', encoding='cp949' )

# print(df1)
# print(df1.shape)
# [660 rows x 16 columns]
# (660, 16)

#비트컴퓨터
df2 = pd.read_csv('./data/csv/비트컴퓨터 1120.csv', header=0, index_col=None, sep=',', encoding='cp949' )

# print(df2)
# print(df2.shape)
# [1200 rows x 16 columns]
# (1200, 16)


# 오름차순 정렬
df1 = df1.sort_values(['일자'], ascending=['True'])
df2 = df2.sort_values(['일자'], ascending=['True'])
print('====오름차순 정렬====')
# print(df1)
# print(df2)

# 삼성전자 시가 고가 저가 금액(백만)x , 종가y 뽑아오기
samsung = df1.loc[626:1,['시가', '고가', '저가', '금액(백만)', '종가']]
# print(samsung)


#삼성전자 모든 데이터 int로 변경
for i in range(len(samsung.index)): # 모든 str -> int 변경
     for j in range(len(samsung.iloc[i])):
        samsung.iloc[i,j] = int(samsung.iloc[i,j].replace(',', ''))

print(samsung)
print(samsung.shape) #(626,5)

#=================================================================

# 비트 시가 고가 저가x , 종가y 뽑아오기
bit = df2.loc[626:1,['시가', '고가', '저가', '종가']]
# print(bit)

#비트 모든 데이터 int로 변경
for i in range(len(bit.index)): # 모든 str -> int 변경
     for j in range(len(bit.iloc[i])):
        bit.iloc[i,j] = int(bit.iloc[i,j].replace(',', ''))

print(bit)
print(bit.shape) #(626,4)

#================================================================

# x, y 데이터 세팅
# 삼성이랑 비트랑 훈련해서 결과는 삼성
# 그러니까 삼성 x,y 비트 x
samsung_x = samsung[['시가', '고가', '저가','금액(백만)']]
samsung_y = samsung[['종가']]

#to numpy 넘파이로 변환
samsung_x = samsung_x.to_numpy()
bit_x = bit.to_numpy()
samsung_y = samsung_y.to_numpy()

#데이터 스케일링 fit은 x데이터만
#삼성
scaler1 = StandardScaler()
scaler1.fit(samsung_x)
samsung_x = scaler1.transform(samsung_x)

#비트 스케일링
scaler2 = StandardScaler()
scaler2.fit(bit_x)
bit_x = scaler2.transform(bit_x)

# x 데이터 다섯개씩 자르기
# 스플릿 함수 임포트 해서 쓰기
size = 5
samsung_x = split_data(samsung_x, size)
bit_x = split_data(bit_x, size)
bit_x = bit_x[:samsung_x.shape[0],:]

# y 데이터 추출
samsung_y = samsung_y[5:, :]

# predict 데이터 추출
# 평가해보자
samsung_x_predict = samsung_x[-1]
bit_x_predict = bit_x[-1]
samsung_x = samsung_x[:-1, :, :]
bit_x=bit_x[:-1, :, :]

samsung_x=samsung_x.astype('float32')
samsung_y=samsung_y.astype('float32')
samsung_x_predict=samsung_x_predict.astype('float32')
bit_x=bit_x.astype('float32')
bit_x_predict=bit_x_predict.astype('float32')

np.save('./data/samsung_x.npy', arr=samsung_x)
np.save('./data/samsung_x_predict.npy', arr=samsung_x_predict)
np.save('./data/samsung_y.npy', arr=samsung_y)
np.save('./data/bit_x.npy', arr=bit_x)
np.save('./data/bit_x_predict.npy', arr=bit_x_predict)

# train, test 분리
samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test=train_test_split(samsung_x, samsung_y, train_size=0.8)
bit_x_train, bit_x_test=train_test_split(bit_x, train_size=0.8)

print("samsung_x.shape : ", samsung_x.shape) #(621, 5, 4)
print("samsung_y.shape : ", samsung_y.shape) #(621, 1)
print("bit_x.shape : ", bit_x.shape) #(621, 5, 4)

samsung_x_predict=samsung_x_predict.reshape(1,5,4)
bit_x_predict=bit_x_predict.reshape(1,5,4)


######### 2. LSTM 회귀모델
samsung_input1=Input(shape=(5,4))
samsung_layer1=LSTM(100, activation='relu')(samsung_input1)
samsunt_layer1=Dropout(0.2)(samsung_layer1)
samsung_layer2=Dense(500, activation='relu')(samsung_layer1)
samsung_layer3=Dense(3000, activation='relu')(samsung_layer2)
samsunt_layer4=Dropout(0.2)(samsung_layer3)
samsung_layer5=Dense(200, activation='relu')(samsunt_layer4)
samsung_layer6=Dense(20, activation='relu')(samsung_layer5)
samsung_layer6=Dense(10, activation='relu')(samsung_layer6)
samsung_output=Dense(1)(samsung_layer6)

bit_input1=Input(shape=(5,4))
bit_layer1=LSTM(30, activation='relu')(bit_input1)
bit_layer2=Dense(200,activation='relu')(bit_layer1)
bit_layer3=Dense(2000,activation='relu')(bit_layer2)
bit_layer3=Dropout(0.2)(bit_layer3)
bit_layer4=Dense(200,activation='relu')(bit_layer3)
bit_layer5=Dense(20,activation='relu')(bit_layer4)
bit_layer6=Dense(10,activation='relu')(bit_layer5)
bit_layer7=Dense(5,activation='relu')(bit_layer6)
bit_output=Dense(1)(bit_layer7)

merge1=concatenate([samsung_output, bit_output])

output1=Dense(300)(merge1)
output2=Dense(3000)(output1)
output3=Dense(800)(output2)
output4=Dense(30)(output3)
output5=Dense(1)(output4)

model=Model(inputs=[samsung_input1, bit_input1], outputs=output5)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es=EarlyStopping(monitor='val_loss',  patience=50, mode='auto')
modelpath='./model/samsung-{epoch:02d}-{val_loss:.4f}.hdf5'
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit([samsung_x_train, bit_x_train], samsung_y_train, epochs=10000, batch_size=1000, validation_split=0.2, callbacks=[es, cp])


#4. 평가, 예측
loss=model.evaluate([samsung_x_test, bit_x_test], samsung_y_test, batch_size=1000)
samsung_y_predict=model.predict([samsung_x_predict, bit_x_predict])

print("loss : ", loss)
print("2020.11.20. 삼성전자 종가 :" , samsung_y_predict)



