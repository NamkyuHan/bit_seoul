#LSTM 확인

from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미 없어요', '너무 재미없다', '참 재밌네요'
]

#1. 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

# 시퀀스의 빈자리를 채우겠다
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre') # 뒤 post
print(pad_x)
print(pad_x.shape)

word_size = len(token.word_index) +1
print('전체 토큰 사이즈 : ', word_size)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten

# 단어 사전의 갯수 아웃풋 갯수 열값
# 인풋 렝스만 틀려도 돌아가긴 하지만 경고가 뜬다 웬만하면 컬럼의 갯수를 잘 보고 맞춰주자
# 단어 사전 갯수는 많아도 되지만 적으면 안돌아간다
model = Sequential()
model.add(Embedding(25,10, input_length=5))
# model.add(Embedding(25,10))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]
print("acc : ", acc)
