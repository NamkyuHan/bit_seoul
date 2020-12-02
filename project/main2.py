import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
import seaborn as sns

import numpy as np

# # 1. hook-fbprophet.py:
# from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# hiddenimports = collect_submodules('fbprophet')
# datas = collect_data_files('fbprophet')
# # 1. hook-pystan.py:
# from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# hiddenimports = collect_submodules('pystan')
# datas = collect_data_files('pystan')
# # 1. hook-Cython.py:
# from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# hiddenimports = collect_submodules('Cython')
# datas = collect_data_files('Cython')


# 전세계 지역별 확진자, 회복자, 사망자 Data Set
url = 'https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv'
data = pd.read_csv(url, error_bad_lines=False)


# Data Set의 구조 파악
print(data.head())

# 코로나 확진자 데이터를 한국 기준으로 날짜, 확진자 순으로 만들기

df_korea = data[data['Country/Region'] == 'Korea, South']

df_korea = np.column_stack([df_korea.Date, df_korea.Confirmed])
df_korea =  pd.DataFrame(df_korea)

df_korea = df_korea.reset_index().rename(columns={0: 'date', 1: 'confirmed'})
df_korea = df_korea.drop("index", axis=1)

print(df_korea.tail())


# 한국 코로나19 확진자 트렌드를 그래프 제작

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_korea.date,
        y=df_korea.confirmed,
        name='Confirmed in Korea'
    )
)

fig.show()


# Facebook Prophet 예측 모델에 넣을 데이터프레임 제작
df_prophet = df_korea.rename(columns={
    'date': 'ds',
    'confirmed': 'y'
})

print(df_prophet.tail())


# Prophet에서 감안할 계절성을 선택해 모델 구성
model = Prophet(
    changepoint_prior_scale=0.2, # 트렌드의 유연성 조절
    changepoint_range=0.9, # 체인지 포인트 설정 가능 범위
    yearly_seasonality=False, #연 계절성
    weekly_seasonality=True, # 주 계절성
    daily_seasonality=True, # 일 계절성
    seasonality_mode='multiplicative' #multiplicative 데이터의 진폭이 증가하거나 감소함  #additive 데이터의 진폭이 일정함
)

# prophet으로 핏, 훈련
model.fit(df_prophet)

future = model.make_future_dataframe(periods=14)
# future_data = pd.DataFrame(['2020-11-01', '2020-11-30'], columns=['ds'])
forecast = model.predict(future)
forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']].tail()

print(forecast.tail())

fig = plot_plotly(model, forecast)
py.iplot(fig)


# display changepoints as red dotted line on the plot.
# changepoint를 그래프에 반영해봅시다.
fig = model.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), model, forecast)






'''
import pkg_resources
model_file = pkg_resources.resource_filename('fbprophet', 'stan_model/prophet_model.pkl')
print(model_file)

import pickle
with open(model_file, 'rb') as f:
    stan_model = pickle.load(f)
'''









