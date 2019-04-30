import pprint
import numpy as np
from knn import KNN, distance_metric, metrics, tt_split, cosine_distance
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Read the text file
fd = open("./iris_data.txt", "r")
lines = fd.readlines()

data = []
label = []
label_dict = {
    "Iris-setosa":1,
    "Iris-versicolor":2,
    "Iris-virginica":3
}

for i in lines:
    temp = list(map(str.strip, i.split(",")))
    data.append(list(map(float, temp[:-1])))
    label.append(label_dict[temp[-1]])

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot(h = .02):
    _data = [[i[0], i[3]] for i in data]
    # Split the data into train and test parts
    ##train_d, train_l, test_d, test_l = tt_split(_data, label)
    train_d, train_l, test_d, test_l = (
        _data[0:30] + _data[50:80] + _data[100:130],
        label[0:30] + label[50:80] + label[100:130],
        _data[30:50] + _data[80:100] + _data[130:],
        label[30:50] + label[80:100] + label[130:]
    )
    # Initialize the KNN object
    knn = KNN(neighbors_num=3, distance=cosine_distance())
    # Fill the data in KNN
    knn.fit(train_d, train_l)

    _t = np.array(train_d)
    x_min, x_max = _t[:, 0].min() - .2, _t[:, 0].max() + .2
    y_min, y_max = _t[:, 1].min() - .2, _t[:, 1].max() + .2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array(knn.predict(Z))
    Z = Z.reshape(xx.shape)
    print(Z)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(_t[:, 0], _t[:, 1], c=train_l, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

def cosine():
    train_d, train_l, test_d, test_l = tt_split(data, label)
    # Initialize the KNN object
    knn = KNN(neighbors_num=5, distance=cosine_distance())
    # Fill the data in KNN
    knn.fit(train_d, train_l)
    # Take prediction from KNN
    result = knn.predict(test_d)

    # Print the results on screen as data, real label, predicted label.
    #print("%20s - %20s | %20s | %s" %("[Data]", "<Real Label>", "<Predicted Label>", "Truth"))

    n = 0
    for i, j, r in zip(test_d, test_l, result):
        truthness = True if j == r else False
        if truthness:
            n += 1
        print("%20s - %20s | %20s | %s" %(i, j, r, truthness))
    print("Acc:", n / len(test_d))
    return n / len(test_d)

def foo(k_num = 5, distance=distance_metric(p=1)):
    _data = [[i[0], i[3]] for i in data]
    # Split the data into train and test parts
    #train_d, train_l, test_d, test_l = tt_split(_data, label)
    train_d, train_l, test_d, test_l = (
        _data[0:30] + _data[50:80] + _data[100:130],
        label[0:30] + label[50:80] + label[100:130],
        _data[30:50] + _data[80:100] + _data[130:],
        label[30:50] + label[80:100] + label[130:]
    )
    # Initialize the KNN object
    knn = KNN(neighbors_num=k_num, distance=distance)
    # Fill the data in KNN
    knn.fit(train_d, train_l)
    # Take prediction from KNN
    result = knn.predict(test_d)

    # Print the results on screen as data, real label, predicted label.
    #print("%20s - %20s | %20s | %s" %("[Data]", "<Real Label>", "<Predicted Label>", "Truth"))

    n = 0
    for i, j, r in zip(test_d, test_l, result):
        truthness = True if j == r else False
        if truthness:
            n += 1
        #print("%20s - %20s | %20s | %s" %(i, j, r, truthness))
    #print("Acc:", n / len(test_d))
    return n / len(test_d), n, len(test_d)

def overall_acc(rounds = 100):
    k_nums = [1, 3, 5, 7, 9, 11, 13, 15]
    distances = [distance_metric(p=1), distance_metric(p=2), cosine_distance()]
    
    for k_num in k_nums:
        print("k=%d" %(k_num), end="")
        for distance in distances:
            a, b, c = foo(k_num=k_num, distance=distance)
            print(" - %2.2f & %d/%d" %(a*100, c - b, c), end="")
        print()
    
    #print("Overall Acc:", nn / rounds)

if __name__ == "__main__":
    #plot(h = .02)
    overall_acc(rounds = 1)
    #cosine()
