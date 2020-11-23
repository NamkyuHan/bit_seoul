import pandas as pd

wine = pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0,)

count_data = wine.groupby('quality')['quality'].count()

print(count_data)
'''
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
'''
import matplotlib.pyplot as plt
count_data.plot()
plt.show()






