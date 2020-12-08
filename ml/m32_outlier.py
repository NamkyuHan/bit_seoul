import numpy as np

# 25~75% 사이의 값들만 데이터로 사용하겠다는 함수
# 그 외의 상단 하단은 제외시킨다
# 다만 각 상단과 하단이 제외되는 만큼 1.5배의 범위만큼 상단 하단을 키워서 사용하겠다
def outliers(data_out):
        quartile_1,  quartile_3 = np.percentile(data_out, [25, 75])
        print("1사분위 : ", quartile_1) #3.25
        print("3사분위 : ", quartile_3) #97.5
        iqr = quartile_3 - quartile_1 # 94.25
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        return np.where((data_out>upper_bound) | (data_out<lower_bound))

a = np.array([1,2,3,4,10000,6,7,5000,90,1000])

b = outliers(a)
print("이상치 위치 : ", b )