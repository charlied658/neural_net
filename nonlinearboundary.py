from random import random
import autograd.numpy as np
from matplotlib import pyplot as plt
from sympy import var, plot_implicit

# Generate coefficient vector
c = [0] * 6

for i in range(6):
    c[i] = random() * 10 - 5

print("c=", c)

# Generate Training Instances
x1 = [0] * 1000
x2 = [0] * 1000
x = []
F = []
for i in range(1000):
    x1[i] = random() * 10 - 5
    x2[i] = random() * 10 - 5
    x.append([1, x1[i], x2[i]])
    f = [1, x1[i], x2[i], x1[i] * x2[i], x1[i] ** 2, x2[i] ** 2]
    F.append(f)

posx = []
posy = []
negx = []
negy = []
y = [0] * 1000
G = []

for i in range(1000):
    if np.dot(F[i], c) > 0:
        posx.append(x1[i])
        posy.append(x2[i])
        y[i] = 1

    else:
        negx.append(x1[i])
        negy.append(x2[i])
        y[i] = -1

var('x y')
plt.scatter(posx, posy, c="green")
plt.scatter(negx, negy, c="red")
plt.title("Nonlinear Decision Boundary")
plot_implicit(c[0] + c[1] * x + c[2] * y + c[3] * x * y + c[4] * x ** 2 + c[5] * y ** 2)
plt.show()

ax = plt.axes(projection='3d')

zdata = np.dot(F, c)
xdata = x1
ydata = x2

ax.scatter3D(xdata, ydata, zdata)
plt.show()
