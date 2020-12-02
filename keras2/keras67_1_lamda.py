# 람다 함수 x에 임의의 숫자를 넣는다 1은 넣으면? 2
gradient = lambda x: 2*x - 4

def gradient2(x) :
    temp = 2*x - 4
    return temp

x = 3    

print(gradient(x)) #2
print(gradient2(x)) #2
