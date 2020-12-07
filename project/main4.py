import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./data/csv/코로나.csv', header=0, index_col=None, sep=',', encoding='cp949' )

df['Date'] = pd.to_datetime(df['Date'])

print(df)
print(df.shape)
print(df.describe)
print(df.info())

df2 = df.loc[df['Country'] == 'Korea, South']
# df2['datetime'] = pd.to_datetime(df2['datetime'])
# pd.Series(pd.to_datetime(['2020-01-22']))
# pd.to_datetime(df2['Date'], format='%y%m%d')


plt.figure(figsize=(16,9))
sns.lineplot(y=df2['Confirmed'], x=df2['Date'])
plt.xlabel('time')
plt.ylabel('CF')
plt.show()

scaler = MinMaxScaler()
scale_cols = ['Confirmed', 'Recovered', 'Deaths']
df2_scaled = scaler.fit_transform(df2[scale_cols])

df2_scaled = pd.DataFrame(df2_scaled)
df2_scaled.columns = scale_cols

print(df2_scaled)
# print(df2)



