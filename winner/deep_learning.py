import tensorflow as tf
from tensorflow import keras
import numpy as np
import csv

def loadData(_x, _y, fileName):
    col_1 = [' State-gov', ' Self-emp-not-inc', ' Private', ' Federal-gov', ' Local-gov', ' ?', ' Self-emp-inc', ' Without-pay', ' Never-worked']
    col_3 = [' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th', ' Some-college', ' Assoc-acdm', ' Assoc-voc', ' 7th-8th', ' Doctorate', ' Prof-school', ' 5th-6th', ' 10th', ' 1st-4th', ' Preschool', ' 12th']
    col_5 = [' Never-married', ' Married-civ-spouse', ' Divorced', ' Married-spouse-absent', ' Separated', ' Married-AF-spouse', ' Widowed']
    col_6 = [' Adm-clerical', ' Exec-managerial', ' Handlers-cleaners', ' Prof-specialty', ' Other-service', ' Sales', ' Craft-repair', ' Transport-moving', ' Farming-fishing', ' Machine-op-inspct', ' Tech-support', ' ?', ' Protective-serv', ' Armed-Forces', ' Priv-house-serv']
    col_7 = [' Not-in-family', ' Husband', ' Wife', ' Own-child', ' Unmarried', ' Other-relative']
    col_8 = [' White', ' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo', ' Other']
    col_9 = [' Male', ' Female']
    col_13 = [' United-States', ' Cuba', ' Jamaica', ' India', ' ?', ' Mexico', ' South', ' Puerto-Rico', ' Honduras', ' England', ' Canada', ' Germany', ' Iran', ' Philippines', ' Italy', ' Poland', ' Columbia', ' Cambodia', ' Thailand', ' Ecuador', ' Laos', ' Taiwan', ' Haiti', ' Portugal', ' Dominican-Republic', ' El-Salvador', ' France', ' Guatemala', ' China', ' Japan', ' Yugoslavia', ' Peru', ' Outlying-US(Guam-USVI-etc)', ' Scotland', ' Trinadad&Tobago', ' Greece', ' Nicaragua', ' Vietnam', ' Hong', ' Ireland', ' Hungary', ' Holand-Netherlands']
    col_14 = [' <=50K', ' >50K', ' <=50K.', ' >50K.']
    with open(fileName) as rawData:
        rows = csv.reader(rawData, delimiter = ",")
        for r in rows:
            if len(r) == 0:
                continue
            temp = []
            for i in range(15):
                if i == 0:
                    temp.append(float(r[i]))
                if i == 1:
                    cnt = 0
                    for c in col_1:
                        if c == r[i]:
                            temp.append(float(cnt))
                            break
                        cnt += 1
                if i == 2:
                    temp.append(float(r[i]))
                if i == 3:
                    cnt = 0
                    for c in col_3:
                        if c == r[i]:
                            temp.append(float(cnt))
                            break
                        cnt += 1
                if i == 4:
                    temp.append(float(r[i]))
                if i == 5:
                    cnt = 0
                    for c in col_5:
                        if c == r[i]:
                            temp.append(float(cnt))
                            break
                        cnt += 1
                if i == 6:
                    cnt = 0
                    for c in col_6:
                        if c == r[i]:
                            temp.append(float(cnt))
                            break
                        cnt += 1
                if i == 7:
                    cnt = 0
                    for c in col_7:
                        if c == r[i]:
                            temp.append(float(cnt))
                            break
                        cnt += 1
                if i == 8:
                    cnt = 0
                    for c in col_8:
                        if c == r[i]:
                            temp.append(float(cnt))
                            break
                        cnt += 1
                if i == 9:
                    cnt = 0
                    for c in col_9:
                        if c == r[i]:
                            temp.append(float(cnt))
                            break
                        cnt += 1
                if i == 10 or i == 11 or i == 12:
                    temp.append(float(r[i]))
                if i == 13:
                    cnt = 0
                    for c in col_13:
                        if c == r[i]:
                            temp.append(float(cnt))
                            break
                        cnt += 1
                if i == 14:
                    cnt = 0
                    for c in col_14:
                        if c == r[i]:
                            _y.append(float(cnt % 2))
                            break
                        cnt += 1
            _x.append(temp)
            
def standardization(dataMatrix):
    if dataMatrix.shape[0] == 0:
        return dataMatrix
    for i in range(dataMatrix.shape[1]):
        sum = 0
        for _x in dataMatrix:
            sum += _x[i]
        mean = sum / dataMatrix.shape[0]
        SD = 0
        for _x in dataMatrix:
            SD += (_x[i] - mean)**2
        SD = np.sqrt(SD / dataMatrix.shape[0])
    
        for _x in dataMatrix:
            _x[i] = (_x[i] - mean) / SD
    return dataMatrix
def meanNormalization(dataMatrix):
    if dataMatrix.shape[0] == 0:
        return dataMatrix
    for i in range(dataMatrix.shape[1]):
        sum = 0
        max = 0
        min = 0
        for data in dataMatrix:
            sum += data[i]
            if data[i] > max:
                max = data[i]
            if data[i] < min:
                min = data[i]
        mean = sum / dataMatrix.shape[0]
        if (max - min) != 0:
            for data in dataMatrix:
                data[i] = (data[i] - mean) / (max - min)
    return dataMatrix            
def rescaling(dataMatrix):
    if dataMatrix.shape[0] == 0:
        return dataMatrix
    for i in range(dataMatrix.shape[1]):
        max = 0
        min = 0
        for data in dataMatrix:
            if data[i] > max:
                max = data[i]
            if data[i] < min:
                min = data[i]
        if max - min != 0:
            for data in dataMatrix:
                data[i] = (data[i] - min) / (max - min)
    return dataMatrix
trainX = []
trainY = []
testX = []
testY = []
loadData(trainX, trainY, 'train.csv')
loadData(testX, testY, 'test.csv')
trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)
trainX = standardization(trainX) #Test accuracy: 0.8504391622392502
testX = standardization(testX)
# trainX = meanNormalization(trainX) #Test accuracy: 0.8516675879786739
# testX = meanNormalization(testX)
# trainX = rescaling(trainX) #Test accuracy: 0.8477366255400303
# testX = rescaling(testX)
model = tf.keras.Sequential([
    keras.layers.Dense(28, activation=tf.nn.relu),
    keras.layers.Dense(14, activation=tf.nn.relu),
    keras.layers.Dense(2,activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(trainX, trainY, batch_size = 64, epochs = 10)
testLoss, testAcc = model.evaluate(testX, testY)
print('Test loss: ', testLoss ,'  Test accuracy:', testAcc)
