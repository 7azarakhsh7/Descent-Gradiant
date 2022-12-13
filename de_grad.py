import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from mpmath import *
from matplotlib import animation


data = np.loadtxt('house_price.txt', delimiter=',')

x = data[:, 0]
y = data[:, 1]


def h(b, a, x):
    return b + a * x


def mse(y_pred, y_true):
    return 0.5 * ((y_pred - y_true) ** 2).mean()


mu = x.mean()
sigma = x.std()
xn = (x - mu) / sigma
# print(xn)

alpha = 5e-3


b = np.random.randn()
a = np.random.randn()

print("Initial guess:")
print(" b= %.4f\n a = %.4f" % (b, a))

costs = []


def update_step():
    global b, a, costs
    y_pred = h(b, a, xn)
    costs.append(mse(y_pred, y))
    db = (y_pred - y)
    da = xn * db
    b -= alpha * db.mean()
    a -= alpha * da.mean()


fig = plt.figure(dpi=100, figsize=(10,6))
plt.scatter(xn, y)
y_pred = h(b, a, xn)
line, = plt.plot(xn, y_pred, 'k')


def animate(i):
    line.set_ydata(h(b, a, xn))
    for i in range(100): update_step()
    return line,


anim = animation.FuncAnimation(fig, animate, np.arange(0, 20), interval=200, repeat_delay=1000)
plt.show(anim)
