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
                    data[(n_row - 1) % 18].append(float(0.0001))
        n_row += 1

x = []
y = []
for i in range(12):
    for j in range(471):
        x.append([])
        for w in range(18):
            for t in range(9):
                x[471 * i + j].append(data[w][480 * i + j + t])
        y.append(data[9][480 * i + j + 9])
x = np.array(x)
y = np.array(y)
x = np.concatenate((np.ones((x.shape[0],1)),x),axis = 1) #在第一列添上一条全为1的列作为bias
w = np.zeros(x.shape[1])
# print(x)
#repeat = 1000;
def lossFunction(target,weight,samples):
    M = target - np.dot(weight,samples.T)
    loss = 0
    for m in M:
        loss += m**2
    return loss

lr = 8
pre_grad = np.zeros(x.shape[1])# 独立的learning rate
loss_rate = []
for r in range(10000):
    temp_loss = 0
    for m in range(36):
        for s in range(156):
            L = np.dot(w,x[157 * m + s].T) - y[157 * m + s] #loss, shape : a number
            grad = np.dot(x[157 * m + s].T,L)*(2)
            pre_grad += grad**2
            ada = np.sqrt(pre_grad)
            w = w - lr * grad/ada
        temp_loss += abs(np.dot(w,x[157 * m + 156].T) - y[157 * m + 156])
    loss_rate.append(temp_loss / 36)
    print("%.2f" % (r * 100 / 10000),'% loss:',"%.4f" % (temp_loss / 36))
plt.plot(loss_rate)
plt.ylabel('loss')
plt.show()
np.save('model.npy',w)
