import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from datetime import datetime, timedelta


# 폰트 설정
plt.style.use('dark_background')
mpl.rcParams['axes.unicode_minus'] = False
font_path = './font/gulim.ttc'
font_name = fm.FontProperties(fname=font_path, size=50).get_name()
plt.rc('font', family=font_name)


# Big 5 데이터셋 로드
pos_questions = [
    'OPN1', 'OPN3', 'OPN5', 'OPN7', 'OPN8', 'OPN9', 'OPN10',            # 7 Openness 개방성
    'CSN1', 'CSN3', 'CSN5', 'CSN7', 'CSN9', 'CSN10',                    # 6 Conscientiousness 성실성
    'EXT1', 'EXT3', 'EXT5', 'EXT7', 'EXT9',                             # 5 Extroversion 외향성
    'AGR2', 'AGR4', 'AGR6', 'AGR8', 'AGR9', 'AGR10',                    # 6 Agreeableness 친화성
    'EST1', 'EST3', 'EST5', 'EST6', 'EST7', 'EST8', 'EST9', 'EST10',    # 8 Emotional Stability 안정성(신경성)
]

neg_questions = [
    'OPN2', 'OPN4', 'OPN6',                                             # 3 Openness 개방성
    'CSN2', 'CSN4', 'CSN6', 'CSN8',                                     # 4 Conscientiousness 성실성
    'EXT2', 'EXT4', 'EXT6', 'EXT8', 'EXT10',                            # 5 Extroversion 외향성
    'AGR1', 'AGR3', 'AGR5', 'AGR7',                                     # 4 Agreeableness 친화성
    'EST2', 'EST4',                                                     # 2 Emotional Stability 안정성(신경성)
]

usecols = pos_questions + neg_questions + ['country']

df = pd.read_csv('./data/csv/data-final.csv', sep='\t', usecols=usecols) #tsv index_col=None

print(len(df)) # 1015341
print(df.head()) # [5 rows x 51 columns]


# Drop Rows Contains 0
# 0점인 칼럼 제외
df = df.replace(0, np.nan).dropna(axis=0).reset_index(drop=True)

print(len(df)) # 874366
print(df.head()) # [5 rows x 51 columns]


# Filter Countries
# 100명 이상이 답한 나라만 선정
df_2 = (df.groupby('country').agg('count')['EXT1'] > 100).reset_index()

fc = df_2[df_2['EXT1'] == True]['country']

df = df[df['country'].isin(fc)].reset_index(drop=True)

print(df) # [872094 rows x 51 columns]


# Positive Negative Scores
# 가운데 점수를 0점으로 조정
df[pos_questions] = df[pos_questions].replace({1:-2, 2:-1, 3:0, 4:1, 5:2})
df[neg_questions] = df[neg_questions].replace({1:2, 2:1, 3:0, 4:-1, 5:-2})

print(df.head()) # [5 rows x 51 columns]


# Compute Scores
# 질문들을 하나의 항목으로 합치기
traits = {
    'OPN' : '개방성',
    'CSN' : '성실성',
    'EXT' : '외향성', 
    'AGR' : '친화성',
    'EST' : '안정성'
}

for trait, trait_kor in traits.items():
    trait_cols = sorted([col for col in df.columns if trait in col])
    df[trait_kor] = df[trait_cols].sum(axis=1)

df_traits = df[list(traits.values()) + ['country']].copy()
print(df_traits.head())
'''
    개방성   성실성   외향성   친화성   안정성 country
0   15.0       2.0    16.0      9.0     -6.0     GB
1    5.0       7.0   -10.0     14.0     -5.0     MY
2   11.0       4.0    -5.0     12.0     -4.0     GB
3    9.0      -5.0    -4.0      8.0     -1.0     GB
4   18.0      18.0    -1.0     16.0    -11.0     KE
'''

# Distribution Plot
fig = plt.figure(figsize=(16, 6))

for trait in traits.values():
    sns.distplot(df_traits[trait], kde=False, bins=40, axlabel=False)

fig.legend(traits.values())    
plt.show()


# Openness VS 한국, 미국 개방성 차이 
fig = plt.figure(figsize=(16, 6))

sns.distplot(df_traits[df_traits['country'] == 'KR']['개방성'], bins=40, axlabel=False)
sns.distplot(df_traits[df_traits['country'] == 'FR']['개방성'], bins=40, axlabel=False)

fig.legend(['한국', '프랑스'])    
plt.show()


# Compute Mean by Countries
# 국가별 성격요소 평균
df_traits_mean = df_traits.groupby('country').mean().rename_axis('country').reset_index()

print(df_traits_mean) #[113 rows x 6 columns]


# COVID-19 Dataset
# 코로나 데이터셋 불러오기
df_covid = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv', 
                        parse_dates=['Date'])

print(df_covid.head())
print(df_covid.tail())


# Country Code
# 국가코드 불러오기
cc = pd.read_csv('./data/csv/country_code.csv')

print(cc.head())


# Filter Dataset Step 1
# 최소 확진자 수 50명 이상
df_covid = df_covid[df_covid['Confirmed'] > 50].reset_index(drop=True) 

df_covid = df_covid.groupby(['Country/Region', 'Date']).sum().reset_index()

print(df_covid[df_covid['Country/Region'] == 'US']) # [269 rows x 5 columns]


# Filter Dataset Step 2
# 50명이 된 후 14일 이후
n_days = 14

filtered = (
    datetime.now() - df_covid.groupby('Country/Region')['Date'].min() > timedelta(days=n_days)
).reset_index().rename(columns = {'Date': 'Filtered'})

filtered_countries = filtered[filtered['Filtered'] == True]['Country/Region']

df_covid = df_covid[df_covid['Country/Region'].isin(filtered_countries)]

df_covid_14days = df_covid.groupby('Country/Region').head(n_days).groupby('Country/Region').tail(1)

print(df_covid_14days) # [180 rows x 5 columns]


# Merge All
# 국가코드 데이터, 성격데이터 병합
df_covid_14days = df_covid_14days.merge(cc, left_on='Country/Region', right_on='Name')

df_covid_14days = df_covid_14days.merge(df_traits_mean, left_on='Code', right_on='country')

print(df_covid_14days.sort_values('Confirmed', ascending=False)) # [96 rows x 13 columns]


# Compute Pearson Correlation
# 피어슨 상관 계수 구하기
new_df = df_covid_14days[
    ~df_covid_14days['country'].isin(['CN', 'TR'])
]


for trait, trait_kor in traits.items():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, n in enumerate(['Confirmed', 'Recovered', 'Deaths']):
        corr = pearsonr(
            new_df[trait_kor],
            new_df[n]
        )

        sns.regplot(x=trait_kor, y=n, data=new_df, ax=axes[i])
        axes[i].set_title('%s, %s, :: r=%.2f, p=%.2f' % (trait_kor, n, corr[0], corr[1]))


# print(new_df)
plt.show()


# Sort by Openness
print("개방성 정렬 결과 : ", new_df.sort_values('개방성', ascending=False))

