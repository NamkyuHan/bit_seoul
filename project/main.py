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


# Big 5 Dataset Load
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
df = df.replace(0, np.nan).dropna(axis=0).reset_index(drop=True)

print(len(df)) # 874366
print(df.head()) # [5 rows x 51 columns]


# Filter Countries
df_2 = (df.groupby('country').agg('count')['EXT1'] > 100).reset_index()

fc = df_2[df_2['EXT1'] == True]['country']

df = df[df['country'].isin(fc)].reset_index(drop=True)

print(df) # [872094 rows x 51 columns]



