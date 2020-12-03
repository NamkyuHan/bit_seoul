import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

df = pd.read_csv('./data/csv/코로나.csv', header=0, index_col=None, sep=',', encoding='cp949' )
# Date

df['Date'] = pd.to_datetime(df['Date'])
print(df)
print(df.shape)
print(df.describe)
print(df.info())


df_korea = df[df['Country'] == 'Korea, South']

# df_korea = np.column_stack([df_korea.Date, df_korea.Confirmed, df_korea.Recovered, df_korea.Deaths])
df_korea = np.column_stack([df_korea.Confirmed, df_korea.Recovered, df_korea.Deaths])
df_korea =  pd.DataFrame(df_korea)

df_korea = df_korea.reset_index().rename(columns={0: 'confirmed', 1: 'Recovered', 2: 'Deaths'})
# df_korea = df_korea.reset_index().rename(columns={0: 'date', 1: 'confirmed', 2: 'Recovered', 3: 'Deaths'})
df_korea = df_korea.drop("index", axis=1)

# print(df_korea)
# print(df_korea.shape)
# print(df_korea.describe)
# [314 rows x 2 columns]
# (314, 2)


df_korea_x = df_korea[['Recovered', 'Deaths']]
df_korea_y = df_korea[['confirmed']]

print(df_korea_x)
print(df_korea_x.shape)
print(df_korea_y)
print(df_korea_y.shape)

df_korea_x = df_korea_x.to_numpy()
df_korea_y = df_korea_y.to_numpy()

print(type(df_korea_x))
print(type(df_korea_y))


# scaler = MinMaxScaler()
# scaler.fit(df_korea_x)
# x = scaler.transform(df_korea_x)

df_korea_x=df_korea_x.astype('float32')
df_korea_y=df_korea_y.astype('float32')


x_train, x_test, y_train, y_test = train_test_split(df_korea_x, df_korea_y, test_size=0.2)
# print(df_korea_x.shape)
# print(df_korea_y.shape)
print("*:*", x_train.shape) # (251, 3)
print("*:*", x_test.shape) # (63, 3)


x_train = x_train.reshape(251, 2, 1)
x_test = x_test.reshape(63, 2, 1)


model=Sequential()
model.add(LSTM(100, activation='relu', input_shape=(2, 1)))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

y_predict=model.predict(x_test)

from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score 
r2=r2_score(y_test, y_predict)
print("R2 : ", r2)





'''
# print(df)
# print(df.shape)
# print(df.describe)
# [59974 rows x 4 columns]
# (59974, 4)

df = df[['Date', 'Country', 'Confirmed']]
print(df)
print(df.shape)
# [59974 rows x 3 columns]
# (59974, 3)

df2 = df[df['Country'] == 'Korea, South']
print(df2)
print(df.shape)
'''









'''
url = 'https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv'
data = pd.read_csv(url, error_bad_lines=False)

print(data.head())


scaler = MinMaxScaler()
scale_cols = ['Confirmed', 'Recovered', 'Deaths']
data = scaler.fit_transform(data[scale_cols])

data = pd.DataFrame(data)
data.columns = scale_cols

print(data)
print(data.describe())



df_korea = data[data['Country/Region'] == 'Korea, South']

df_korea = np.column_stack([df_korea.Date, df_korea.Confirmed])
df_korea =  pd.DataFrame(df_korea)

df_korea = df_korea.reset_index().rename(columns={0: 'date', 1: 'confirmed'})
df_korea = df_korea.drop("index", axis=1)

print(df_korea.tail())
'''

