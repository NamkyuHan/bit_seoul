from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten
#Flatten?
# CNN에서 컨볼루션 레이어나 맥스풀링 레이어를 반복적으로 거치면 주요 특징만 추출되고, 추출된 주요 특징은 전결합층에 전달되어 학습됩니다. 
# 컨볼루션 레이어나 맥스풀링 레이어는 주로 2차원 자료를 다루지만 전결합층에 전달하기 위해선 1차원 자료로 바꿔줘야 합니다. 
# 이 때 사용되는 것이 플래튼 레이어입니다. 사용 예시는 다음과 같습니다.
# Flatten()

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1))) #여길 통과하면 (10,10,1) 이게 (9,9,10)으로 증가한다 실질적인 크기는 줄었지만 양이 늘었다 왜 
model.add(Conv2D(5, (2,2), padding='same')) #same 때문에 그대로 전달해줘서 (9,9,5) 주어진 필터 갯수로 다시 설정된다 
model.add(Conv2D(3, (3,3), padding='valid')) # valid 때문에 (9,9,5)로 전달 (7,7,3) 주어진 필터 갯수로 다시 설정된다
model.add(Conv2D(7, (2,2))) # valid 때문에 (7,7,3)로 전달 (6,6,7) 주어진 필터 갯수로 다시 설정된다
model.add(MaxPooling2D()) #(3,3,7)
model.add(Flatten()) #(3*3*7) = 63 노드의 갯수가 63개
model.add(Dense(1)) #최종 아웃풋

model.summary()
# (인풋 x 커널사이즈 + 바이어스) x 필터개수
# (1x2x2+1)x10 = 50
# (10x2x2+1)x5 = 205
# (5x3x3+1)x3 = 138
# (3x2x2+1)x7 = 91


# kernel_size : (2,2) 두칸씩 잘랐다 (3,3) 세칸씩 잘랐다
# filters : 필터의 개수를 지정-여기서는 10개(이건 내가 수정 가능)
# kernel_size : convolution window의 너비와 높이(height and width)를 지정 여기서는 (2,2)
# strides : convolution의 stride를 지정(디폴트 = 1)
# padding : 'valid' 또는 'same' 을 지정(디폴트 = 'valid') 패딩을 씌우면 동일한 쉐이프로 던져준다 데이터의 손실을 막고 공평성을 주기 위해 쓴다
# 입력모양 : batch_size, rows, cols, channels = 여기서는 batch_size 사진 몇장씩 잘라서 작업할래?(잘라서 작업하는 갯수) (5,5,1)
# input_shape : input의 높이, 너비 및 깊이를 지정해주는 튜플

# 참고 LSTM
# units? 이게 인풋 사이즈임
# return_sequence 넘겨주는 차원을 설정
# 입력모양 : batch_size, timesteps, freature = 여기서는 batch_size 몇개씩 잘라서 작업할래?(잘라서 작업하는 갯수) (3,3)
# input_shape = (timesteps, freature) (3,3)예시











