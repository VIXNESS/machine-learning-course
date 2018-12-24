import csv
import numpy as np
from numpy.linalg import inv
import math
import random
import sys
import matplotlib.pyplot as plt

test_row = 0
test_data = []
    
with open('test.csv','r',encoding = 'big5') as text:
    row = csv.reader(text, delimiter = ",")
    for r in row:
        if test_row % 18 == 0:
            test_data.append([])
            for i in range(2,11):
                test_data[test_row // 18].append(float(r[i]))
        else:
            for i in range(2,11):
                if r[i] != "NR":
                    test_data[test_row // 18].append(float(r[i]))
                else:
                    test_data[test_row // 18].append(float(0.00001))  
        test_row += 1

x = []
cur = 0
for td in test_data:
    x.append([])
    for i in range(18):
        if i == 4 or i == 6 or i ==8 or i == 9 or i == 10 or i == 12:
            for j in range(9):
                x[cur].append(td[9 * i + j])
    cur += 1

x = np.array(x)
x = np.concatenate((np.ones((x.shape[0],1)),x),axis = 1)

y = []
rr = 0
with open('ans.csv','r',encoding = 'big5') as ans:
    row = csv.reader(ans,delimiter = ',')
    for r in row:
        if rr != 0:
            y.append(float(r[1]))
        rr += 1

y = np.array(y)
w = np.load("m2.npy")
t = np.dot(x,w)
L = t - y
loss = []
sum = 0
for l in L:
    loss.append(abs(l))
    # print("loss: ",l)
    sum += abs(l)
print(sum / len(L))
plt.plot(y,color = "red")
plt.plot(t,color = "blue")
plt.ylabel('pm 2.5')
plt.show()