from random import random
import autograd.numpy as np
from matplotlib import pyplot as plt


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Generate coefficient vector
c = [0] * 6

for i in range(6):
    c[i] = random() * 10 - 5

print("c=", c)

# Generate Training Instances
x1 = [0] * 100
x2 = [0] * 100
x = []
F = []
for i in range(100):
    x1[i] = random() * 10 - 5
    x2[i] = random() * 10 - 5
    x.append([1, x1[i], x2[i]])
    f = [1, x1[i], x2[i], x1[i] * x2[i], x1[i] ** 2, x2[i] ** 2]
    F.append(f)

y = [0] * 100

# Label instance vectors
for i in range(100):
    if np.dot(F[i], c) > 0:
        y[i] = 1

    else:
        y[i] = -1

# Generate neural net weights
# w1: input -> hidden layer 1 (3x3)
# w2: hidden layer 1 -> hidden layer 2 (4x3)
# w3: hidden layer 2 -> output (4x1)
w1 = []
w2 = []
w3 = []

for i in range(3):
    a = []
    for j in range(3):
        a.append(random() * 10 - 5)
    w1.append(a)

for i in range(4):
    a = []
    for j in range(3):
        a.append(random() * 10 - 5)
    w2.append(a)

for i in range(4):
    w3.append(random() * 10 - 5)

print("w1=", np.array(w1))
print("w2=", np.array(w2))
print("w3=", np.array(w3))
print()

# Activation functions
# a1: activation for hidden layer 1
# a2: activation for hidden layer 2
# a3: activation for output
a1 = [0] * 4
a2 = [0] * 4
a3 = 0

# Gradients
# g1: gradient for hidden layer 1
# g2: gradient for hidden layer 2
# g3: gradient for output
g1 = [0] * 4
g2 = [0] * 4
g3 = 0

print("Training Neural Net")
print()

# Train neural net
for x_ii in range(100):
    for x_i in range(100):
        x0 = x[x_i]

        # Forward pass

        # Input layer -> Hidden layer 1
        for i in range(3):
            net = 0
            for j in range(3):
                net += x0[j] * w1[j][i]
            a1[i] = sigmoid(net)
        a1[3] = 1  # Bias term

        # Hidden layer 1 -> Hidden layer 2
        for i in range(3):
            net = 0
            for j in range(4):
                net += a1[j] * w2[j][i]
            a2[i] = sigmoid(net)
        a2[3] = 1  # Bias term

        # Hidden Layer 2 -> Output
        net = 0
        for i in range(4):
            net += a2[i] * w3[i]
        a3 = sigmoid(net)

        # Backward pass

        # Step size
        a = 0.1

        # Loss for output
        g3 = a3 - (y[x_i] + 1)/2

        # Loss for hidden layer 2
        for i in range(4):
            g2[i] = g3 * w3[i] * a2[i] * (1 - a2[i])

        # Loss for hidden layer 1
        for i in range(4):
            sum = 0
            for j in range(3):
                sum += g2[j] * w2[i][j]
            g1[i] = sum * a1[i] * (1 - a1[i])

        # Update weights from hidden layer 2 -> output
        for i in range(4):
            w3[i] -= a * g3 * a2[i]

        # Update weights from hidden layer 1 -> hidden layer 2
        for i in range(4):
            for j in range(3):
                w2[i][j] -= a * g2[j] * a1[i]

        # Update weights from input layer -> hidden layer 1
        for i in range(3):
            for j in range(3):
                w1[i][j] -= a * g1[j] * x0[i]

print("w1=", np.array(w1))
print("w2=", np.array(w2))
print("w3=", np.array(w3))

# Generate Test Instances
x1 = [0] * 100
x2 = [0] * 100
x = []
F = []
for i in range(100):
    x1[i] = random() * 10 - 5
    x2[i] = random() * 10 - 5
    x.append([1, x1[i], x2[i]])
    f = [1, x1[i], x2[i], x1[i] * x2[i], x1[i] ** 2, x2[i] ** 2]
    F.append(f)

posx = []
posy = []
negx = []
negy = []
y = [0] * 100

for i in range(100):
    if np.dot(F[i], c) > 0:
        posx.append(x1[i])
        posy.append(x2[i])
        y[i] = 1

    else:
        negx.append(x1[i])
        negy.append(x2[i])
        y[i] = -1

misy = []
misx = []
misclassified_instances = 0

# Count misclassified
for x_i in range(100):
    x0 = x[x_i]

    # Forward pass

    # Input layer -> Hidden layer 1
    for i in range(3):
        net = 0
        for j in range(3):
            net += x0[j] * w1[j][i]
        a1[i] = sigmoid(net)
    a1[3] = 1  # Bias term

    # Hidden layer 1 -> Hidden layer 2
    for i in range(3):
        net = 0
        for j in range(4):
            net += a1[j] * w2[j][i]
        a2[i] = sigmoid(net)
    a2[3] = 1  # Bias term

    # Hidden Layer 2 -> Output
    net = 0
    for i in range(4):
        net += a2[i] * w3[i]
    a3 = sigmoid(net)

    # Detect misclassified instance
    if y[x_i] * (2 * a3 - 1) < 0:
        misclassified_instances += 1
        misy.append(x2[x_i])
        misx.append(x1[x_i])

print("Misclassified:", misclassified_instances)

# Plot training set
plt.ylim(5, -5)
plt.xlim(5, -5)
plt.grid(True, which='both')
plt.scatter(posx, posy, c="green")
plt.scatter(negx, negy, c="red")
plt.scatter(misx, misy, c="yellow", s=10)
plt.title("Neural Net")
plt.show()
