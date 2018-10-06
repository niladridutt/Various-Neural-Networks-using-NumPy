import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0],[1,0],[0,1],[1,1]],float)
y = np.array([[0],[1],[1],[0]])

m = X.shape[0]
n = X.shape[1]
hidden = 8
alpha = 0.03

def sigmoid(z,deriv = False):
    sig = 1./(1+np.exp(-z))

    if deriv:
        return sig*(1-sig)
    else:
        return sig

theta1 = np.random.randn(n,hidden)
theta2 = np.random.randn(hidden,y.shape[1])

def forwardProp(X,theta1,theta2):

    a1 = X
    z1 = a1.dot(theta1)
    a2 = sigmoid(z1)
    z2 = a2.dot(theta2)
    hyp = sigmoid(z2)

    return a1,z1,a2,z2,hyp

a1,z1,a2,z2,hyp = forwardProp(X,theta1,theta2)

def get_cost(hyp,y):

    pos = (y*(np.log(hyp)))
    neg = ((1-y)*(np.log(1-hyp)))

    return (-1./m)*np.sum((pos+neg))

cost = list()
def backProp(X,theta1,theta2):

    for i in range(50000):

        a1,z1,a2,z2,hyp = forwardProp(X,theta1,theta2)

        cost.append(get_cost(hyp,y))

        dz2 = hyp-y
        dw2 = a2.T.dot(dz2*sigmoid(z2,True))
        dz1 = (dz2.dot(theta2.T)*sigmoid(z1,True))
        dw1 = a1.T.dot(dz1)

        theta1 += -alpha*dw1
        theta2 += -alpha*dw2

    return theta1,theta2

theta1,theta2 = backProp(X,theta1,theta2)
a1,z1,a2,z2,hyp = forwardProp(X,theta1,theta2)
print(hyp)

y_axis = np.array(cost)
x_axis = np.array([i for i in range(50000)])
plt.scatter(x_axis,y_axis)
plt.show()
