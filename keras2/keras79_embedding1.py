from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 맛있는 밥을 진짜 먹었다'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# {'나는': 1, '울트라': 2, '맛있는': 3, '밥을': 4, '먹었다': 5}
print(len(token.word_index)) # 5

# {'울트라': 1, '나는': 2, '맛있는': 3, '밥을': 4, '먹었다': 5} 울트라 2번 썼을때
# {'진짜': 1, '나는': 2, '맛있는': 3, '밥을': 4, '먹었다': 5} 진짜 2번 썼을때

x = token.texts_to_sequences([text])
print(x) # [[2, 1, 3, 4, 1, 5]] 매칭된 번호 순서대로 출력한다

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
x = to_categorical(x, num_classes=word_size + 1)
print(x)
'''
앞에 0은 원핫이라 다 붙는것 딱히 상관 안해도 됨
[[[0. 0. 1. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0.]
  [0. 0. 0. 1. 0. 0.]
  [0. 0. 0. 0. 1. 0.]
  [0. 1. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 1.]]]
'''









