import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape) #(442, 10)
print(y.shape) #(442, )

# pca = PCA(n_components=9) # 내가 출력하고 싶은 열을 설정할 수 있다 차원축소
# x2d = pca.fit_transform((x))
# print(x2d.shape) #(442, 9)
'''
기존 차원보다 높게 설정하면 이런 오류가 난다
ValueError: n_components=11 must be between 0 and min(n_samples, n_features)=10 with svd_solver='full'
'''

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR) #1.
# print(sum(pca_EVR)) #2.

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

d = np.argmax(cumsum >= 0.95) +1
print(cumsum >= 0.95)
print(d)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()


'''
1. pca_EVR
내가 설정한 컴포넌트 개수대로 나온다
[0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192
 0.05365605 0.04336832 0.00783199]
'''
'''
2. sum(pca_EVR) 숫자가 높을 수록 좋다
0.9991439470098975
'''

