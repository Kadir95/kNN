import numpy as np
import random

class distance_metric:
    def __init__(self, p=1):
        self.p = float(p)
    
    def distance(self, a, b):
        return np.power(np.sum(np.abs(np.power(np.subtract(a, b), self.p))), 1 / self.p)

class cosine_distance:
    def distance(self, a, b):
        return -(np.inner(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))

class KNN:
    def __init__(self, neighbors_num = 5, distance = distance_metric(p=2)):
        self.distace = distance
        self.nn = neighbors_num

    def fit(self, data, label):
        self.data = data
        self.label = label
    
    def predict(self, data):
        results = []
        for i in data:
            d = {}
            t = self._find_nearest(i)
            for j in t:
                if d.get(j[1]) is None:
                    d[j[1]] = 1
                else:
                    d[j[1]] += 1
            maximum = 0
            maximum_i = None
            for k in list(d.keys()):
                if d[k] > maximum:
                    maximum = d[k]
                    maximum_i = k
            results.append(maximum_i)
        return results

    def _find_nearest(self, point):
        results = []
        for i in range(0, self.nn):
            minimum = (self.data[0], self.label[0])
            for i in range(1, len(self.data)):
                if (self.data[i], self.label[i]) not in results and self.distace.distance(minimum[0], point) > self.distace.distance(point, self.data[i]):
                    minimum = (self.data[i], self.label[i])
            results.append(minimum)
        return results

def tt_split(data, label, p=.3):
    l = list(zip(data, label))
    random.shuffle(l)

    cut_point = int(len(l) * p)
    
    train = l[cut_point:]
    test = l[:cut_point]

    train_d = [i[0] for i in train]
    train_l = [i[1] for i in train]
    test_d = [i[0] for i in test]
    test_l = [i[1] for i in test]
    return train_d, train_l, test_d, test_l

metrics = {
    "manhattan": 1,
    "euclidean": 2
}

if __name__ == "__main__":
    a = np.array([1, 2])
    b = np.array([0, 1])
    metr = cosine_distance()
    print(metr.distance(a, b))