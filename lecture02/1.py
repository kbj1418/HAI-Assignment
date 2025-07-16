import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([63,68,76,83,89,94,97,99,100,100])

m = 0.0
c = 0.0
alpha = 0.01
epochs = 1000

n = len(x)
loss_history = []

for i in range(epochs):
    y_pred = m*x+c
    error = y-y_pred

    loss = np.mean(error ** 2)
    loss_history.append(loss)

    dm = -2 * np.mean(x * error)
    dc = -2 * np.mean(error)

    m -= alpha * dm
    c -= alpha * dc
    print("기울기m =",m)
    print("절편c =",c)
    print("학습된 모델: y =",m,"x +",c)