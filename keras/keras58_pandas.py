import numpy as np
import pandas as pd

datasets = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0, sep=',' )

print(datasets) 

print(datasets.shape) 

#######확인해보기#######
# index_col = None, 0 , 1 / header = None, 0, 1
# None은 너 인덱스랑 헤더가 없네? 내가 만들어 줄게
# 0은 너 인덱스랑 헤더가 1줄이구나
# 1은 너 인덱스랑 헤더가 2줄이구나
# 더 추가도 가능하다 데이터를 보고 판단하는게 중요함

# index_col = None
# header = None
# result = [151 rows x 6 columns], (151, 6)
'''
         0             1            2             3            4        5
0      NaN  sepal_length  sepal_width  petal_length  petal_width  species
2      2.0           4.9            3           1.4          0.2        0
3      3.0           4.7          3.2           1.3          0.2        0
4      4.0           4.6          3.1           1.5          0.2        0
..     ...           ...          ...           ...          ...      ...
146  146.0           6.7            3           5.2          2.3        2
147  147.0           6.3          2.5             5          1.9        2
148  148.0           6.5            3           5.2            2        2
149  149.0           6.2          3.4           5.4          2.3        2
150  150.0           5.9            3           5.1          1.8        2
'''
# index_col = 0
# header = 0
# result = [150 rows x 5 columns], (150, 5)
'''
sepal_length  sepal_width  petal_length  petal_width  species
1             5.1          3.5           1.4          0.2        0
2             4.9          3.0           1.4          0.2        0
3             4.7          3.2           1.3          0.2        0
4             4.6          3.1           1.5          0.2        0
5             5.0          3.6           1.4          0.2        0
..            ...          ...           ...          ...      ...
146           6.7          3.0           5.2          2.3        2
147           6.3          2.5           5.0          1.9        2
148           6.5          3.0           5.2          2.0        2
149           6.2          3.4           5.4          2.3        2
150           5.9          3.0           5.1          1.8        2
'''
# index_col = 1 
# header = 1
# result = [149 rows x 5 columns], (149, 5)
'''
      1  3.5  1.4  0.2  0
5.1
4.9    2  3.0  1.4  0.2  0
4.6    4  3.1  1.5  0.2  0
5.0    5  3.6  1.4  0.2  0
5.4    6  3.9  1.7  0.4  0
..   ...  ...  ...  ... ..
6.7  146  3.0  5.2  2.3  2
6.3  147  2.5  5.0  1.9  2
6.5  148  3.0  5.2  2.0  2
6.2  149  3.4  5.4  2.3  2
5.9  150  3.0  5.1  1.8  2
'''
print(datasets.head()) # head는 위에서부터 5개 보여줌
print(datasets.tail()) # tail은 밑에서부터 5개 보여줌
print(type(datasets))

aaa = datasets.values # 판다스를 넘파이로 바꾸는 명령문
print(type(aaa))
print(aaa.shape)

# np.save('./data/iris_ys_pd.npy', arr=aaa)