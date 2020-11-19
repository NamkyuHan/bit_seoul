#실습
#iris_ys2.csv 파일을 넘파이로 불러오기

#불러온 데이터를 판다스로 저장하시오
#(150,5)

#모델완성
#x,y가 붙어 있으므로 슬라이싱 필요함

import numpy as np
import pandas as pd

datasets = np.loadtxt('./data/csv/iris_ys2.csv', delimiter=",")
# datasets = pd.read_csv('./data/csv/iris_ys2.csv', header=None, index_col=None, sep=',' )

print(datasets) # 데이터 확인
print(datasets.shape) # (150, 5)
print(type(datasets)) # 넘파이 클래스 변환 확인 <class 'numpy.ndarray'>



# datasets.to_csv('./data/csv/iris_ys2_pd.csv', header=None, index_col=None, sep=',')

# print(datasets) 
# print(datasets.shape) 
# print(type(datasets))
