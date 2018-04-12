import numpy as np

Dist = np.array([[0, 206, 429, 1504, 963, 2976, 3095, 2979, 1949],
                 [206, 0, 233, 1308, 802, 2815, 2934, 2786, 1771],
                 [429, 233, 0, 1075, 671, 2684, 2799, 2631, 1616],
                 [1504, 1308, 1075, 0, 1329, 3273, 3053, 2687, 2037],
                 [963, 802, 671, 1329, 0, 2013, 2142, 2054, 996],
                 [2976, 2815, 2684, 3273, 2013, 0, 808, 1131, 1307],
                 [3095, 2934, 2799, 3053, 2142, 808, 0, 379, 1235],
                 [2979, 2786, 2631, 2687, 2054, 1131, 379, 0, 1059],
                 [1949, 1771, 1616, 2037, 996, 1307, 1235, 1059, 0]])

cities = np.random.randint(1000, size=(9, 2))


def distance(x1, x2):
    return np.linalg.norm(x1 - x2)


def display(x, D):
    n = D.shape[0]
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[i, j] = distance(x[i], x[j])
    # return (abs((D - d)).astype(int))
    return (abs((D - d)))


display(cities, Dist)


def derivative(x, D, i):
    res = np.zeros((1, 2))
    n = D.shape[0]
    for j in range(n):
        if i == j:
            continue
        d = distance(x[i], x[j])
        res += ((d - D[i, j]) / d) * (x[i] - x[j])
    return 2. * res


def update(x, D):
    n, m = x.shape[0], x.shape[1]
    dx = np.zeros((n, m))
    for i in range(n):
        dx[i] = 0.01 * derivative(x, D, i)
    return x - dx


def get_score(x, D):
    n = x.shape[0]
    res = 0
    for i in range(n):
        for j in range(i):
            res += (distance(x[i], x[j]) - D[i, j]) ** 2
    return res


cities = np.random.randint(1000, size=(9, 2))
get_score(cities, Dist)

for i in range(1000):
    cities = update(cities, Dist)
    if not i % 10:
        print("epoch: {}, score: {}".format(i, get_score(cities, Dist)))

get_score(cities, Dist)
err = display(cities, Dist)
np.seterr(divide='ignore', invalid='ignore')
err_portion = np.divide(err.astype(float), Dist.astype(float))
err_portion.mean()
err


cities
