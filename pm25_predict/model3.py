import csv
import numpy as np
from numpy.linalg import inv
import math
import random
import sys
import matplotlib.pyplot as plt

data = []
for i in range(18):
    data.append([])
n_row = 0
with open('train.csv','r',encoding = 'big5') as text:
    row = csv.reader(text, delimiter = ",")
    for r in row:
        if n_row != 0:
            for i in range(3,27):
                if r[i] != "NR":
                    data[(n_row - 1) % 18].append(float(r[i]))
                else:
                    data[(n_row - 1) % 18].append(float(0.001))
        n_row += 1

x = []
y = []
for i in range(12):
    for j in range(471):
        x.append([])
        for k in range(18):
            if k == 9 or k == 10:
                for m in range(9):
                    x[471 * i + j].append(data[k][480 * i + j + m])
        y.append(data[9][480 * i + j + 9])

x = np.array(x)
y = np.array(y)
x = np.concatenate((np.ones((x.shape[0],1)),x),axis = 1)
w = np.zeros(x.shape[1])

lr = 10
repeat = 10000
pre_grad = np.zeros(x.shape[1])
loss_rate = []
for i in range(repeat):
    loss = 0
    for j in range(36):
        for k in range(156):
            L = np.dot(w,x[157 * j + k].T) - y[157 * j + k]
            grad = 2 * np.dot(x[157 * j + k].T,L)
            pre_grad += grad**2
            ada = np.sqrt(pre_grad)
            w = w - lr * grad / ada
        loss += abs(np.dot(w,x[157 * j + 156].T) - y[157 * j + 156])
    loss_rate.append(loss / 36)
    print("%.2f" % (i/repeat * 100),'% loss:',"%.4f" % (loss / 36))

plt.plot(loss_rate)
plt.show()

np.save('m3.npy',w)