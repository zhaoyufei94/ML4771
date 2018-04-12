import numpy as np
np.seterr(divide='ignore', invalid='ignore')

Dist = np.array([[0, 206, 429, 1504, 963, 2976, 3095, 2979, 1949],
                 [206, 0, 233, 1308, 802, 2815, 2934, 2786, 1771],
                 [429, 233, 0, 1075, 671, 2684, 2799, 2631, 1616],
                 [1504, 1308, 1075, 0, 1329, 3273, 3053, 2687, 2037],
                 [963, 802, 671, 1329, 0, 2013, 2142, 2054, 996],
                 [2976, 2815, 2684, 3273, 2013, 0, 808, 1131, 1307],
                 [3095, 2934, 2799, 3053, 2142, 808, 0, 379, 1235],
                 [2979, 2786, 2631, 2687, 2054, 1131, 379, 0, 1059],
                 [1949, 1771, 1616, 2037, 996, 1307, 1235, 1059, 0]])

class City:
    def __init__(self, size=(9, 2), D=Dist):
        self.cities = np.random.randint(1000, size=size).astype(float)
        self.Dist = D
        self.n, self.m = size[0], size[1]
        print("initial score: {}".format(self.get_score()))

    def distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def get_error(self):
        d = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                d[i, j] = self.distance(self.cities[i], self.cities[j])
        return (abs(self.Dist - d))

    def get_score(self):
        res = 0
        for i in range(self.n):
            for j in range(i):
                res += (self.distance(self.cities[i], self.cities[j]) -
                self.Dist[i, j]) ** 2
        return res

    def derivative(self, i):
        res = np.zeros((1, self.m))
        for j in range(self.n):
            if i == j:
                continue
            d = self.distance(self.cities[i], self.cities[j])
            res += ((d - self.Dist[i, j]) / d) * (self.cities[i] - self.cities[j])
        return res

    def update(self):
        delta = np.zeros((self.n, self.m))
        for i in range(self.n):
            delta[i] = 0.01 * self.derivative(i)
        self.cities -= delta

city = City()


for i in range(2000):
    city.update()
    if not i % 50:
        print("epoch: {}, score: {}".format(i, city.get_score()))

print("distance error")
print(city.get_error())
err = city.get_error()
err_portion = np.divide(err.astype(float), city.Dist.astype(float))
for i in range(9):
    err_portion[i, i]=0
print("portional error")
print(err_portion)
for i in range(9):
    err_portion[i, i]=0
print("mean protional error: {}".format(err_portion.mean()))

print("city coordinates:")
print(city.cities)


import matplotlib.pyplot as plt
def plot():
    plt.scatter(city.cities[0, 0], city.cities[0, 1], c='m', label='BOS')
    plt.scatter(city.cities[1, 0], city.cities[1, 1], marker='o', c='r', label='NYC')
    plt.scatter(city.cities[2, 0], city.cities[2, 1], marker='x', c='c', label='DC')
    plt.scatter(city.cities[3, 0], city.cities[3, 1], c='y', label='MIA')
    plt.scatter(city.cities[4, 0], city.cities[4, 1], c='b', label='CHI')
    plt.scatter(city.cities[5, 0], city.cities[5, 1], c='g', label='SEA')
    plt.scatter(city.cities[6, 0], city.cities[6, 1], c='k', label='SF')
    plt.scatter(city.cities[7, 0], city.cities[7, 1], c='c', label='LA')
    plt.scatter(city.cities[8, 0], city.cities[8, 1], marker='x', c='m', label='DEN')
    plt.legend()
    plt.show()

plot()
