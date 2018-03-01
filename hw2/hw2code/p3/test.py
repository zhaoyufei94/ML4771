n = []
n.append([1, 1, 1, 1])
n.append([2, 2, 1, 1])
n.append([1, 2, 1, 1])
n.append([2, 1, 1, 1])
n.append([4, 4, 1, -1])
n.append([5, 5, 1, -1])
n.append([4, 5, 1, -1])
n.append([5, 4, 1, -1])

w = [0, 0, 0]

def cal(n, w):
    cal = 0
    for i in range(3):
        cal += n[i]*w[i]
    return cal * n[3]

def add(n, w):
    for i in range(3):
        w[i] += n[i] * n[3]
    return w


while True:
    flag = 0
    for i in range(7, -1, -1):
    #for i in range(3, -1, -1):
        if cal(n[i], w) <= 0:
            w = add(n[i], w)
            flag = 1
    if flag == 1:
        continue
    break

print(w)