'''
def split_x(seq, size):
    aaa = [] # 임시 리스트
    # i는 0부터 seq사이즈-size까지 반복 
    # (그래야 size만큼씩 온전히 자를 수 있다)
    for i in range(len(seq) -size +1 ):
        subset = seq[i:(i+size)] # subset은 i부터 size만큼 배열 저장
        aaa.append([subset]) # 배열에 subset을 붙인다
    print(type(aaa)) # aaa의 타입은 리스트
    return np.array(aaa) # 리스트를 어레이로 바꿔서 반환하자
'''

import numpy as np 

# 2차원 리스트의 행을 사이즈만큼 잘라 3차원으로 반환하는 split함수
def split_data(x, size) :
    data=[]
    for i in range(x.shape[0]-size+1) :
        data.append(x[i:i+size,:])
    return np.array(data)



# 테스트 : 2차원 데이터셋
from sklearn.datasets import load_iris
dataset=load_iris()
x=dataset.data #(150,4)
y=dataset.target #(150,)
size=5
# (5,4) 행렬 데이터 146개

print(x)
print(x.shape, y.shape) #(150, 4) (150,)




data_iris=split_data(x, size)
print('======================')
print(data_iris)
print(data_iris.shape) #(146, 5, 4)