import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu(x)

print("x : ", x)
print("y : ", y)

plt.plot(x, y)
plt.grid()
plt.show()

# relu 친구들 찾기 
# elu, selu

'''
####ReLU(rectified linear unit)###
특징 : 0이하의 값음 다음 레이어에 전달하지 않습니다. 0 이상의 값은 그대로 출력합니다
사용처 : CNN을 학습시킬 때 많이 사용됩니다
한계점 : 한번 0 활성화 값을 다음 레이어에 전달하면 이후의 뉴런들의 출력값이 모두 0이 되는 현상이 발생합니다 이를 dying relu라 부릅니다
이러한 한계점을 개선하기 위해 음수 출력 값을 소량이나마 다음 레이어에 전달하는 방식으로 개선한 활성화 함수들이 등장합니다

###LeakyReLU###
특징 : relu와 거의 비슷한 형태를 갖습니다 입력값이 음수일 때 완만한 선형 함수를 그려줍니다 
일반적으로 알파를 0.01로 설정합니다 
사용처: Rectifier Nonlinearities Improve Neural Network Acoustic Models

###ELU(exponential linear unit)###
특징 : relu와 거의 비슷한 형태를 갖습니다 지수 함수를 이용하여 입력이 0 이하일경우 부드럽게 깎아줍니다
미분 함수가 끊어지지 않고 이어져있는 형태를 보입니다 별도의 알파 값을 파라미터로 받는데 일반적으로 1로 설정됩니다
그 밖의 값을 가지게 될 경우 SeLU(scaled exponential linear unit)이라 부릅니다. 알파를 2로 설정할 경우 그래프는 아래와 같은 모습을 보입니다.
사용처: Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)

###PReLU(parametric rectified linear unit)###
특징: LeakyReLU와 거의 유사한 형태를 보입니다. 하지만 LeakyReLU에서는 알파 값이 고정된 상수였던 반면에 PReLU에서는 학습이 가능한 파라미터로 설정됩니다.
사용처: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

###ThresholdReLU(thresold rectified linear unit)###
특징: ReLU와 거의 유사한 형태를 보입니다. ReLU가 0 이하의 입력 값에 대해 0을 출력했다면 ThresoldReLU는 그 경계값을 설정할 수 있으며, 1을 기본값으로 설정합니다.
사용처: Zero-Bias Autoencoders and the Benefits of Co-Adapting Features
'''