import numpy as np
import sklearn,sklearn.datasets

# create dataset
na = numpy.newaxis
Xd,Td = sklearn.datasets.make_moons(n_samples=100)
Xd = Xd*2-1
Td = Td * 2 - 1
X1d = Xd[:,0]
X2d = Xd[:,1]

# Creates a grid dataset on which to inspect the decision function
l = numpy.linspace(-4,4,100)
X1g,X2g = numpy.meshgrid(l,l)

# plotting
def plot(Yg,title=None):
    plt.figure(figsize=(3,3))
    plt.scatter(*Xd[Td==-1].T,color='#0088FF')
    plt.scatter(*Xd[Td==1].T,color='#FF8800')
    plt.contour(X1g,X2g,Yg,levels=[0],colors='black',linestyles='dashed')
    plt.contourf(X1g,X2g,Yg,levels=[-100,0,100],colors=['#0088FF','#FF8800'],alpha=0.1)
    if title is not None: plt.title(title)
    plt.show()

# 50 neurons, neural network eq:
# z_j = x_1w_1j + x_2ww_2j+b_j
# a_j = max(0,z_j)
# y = sum(a_jv_j)

NH = 50
W = numpy.random.normal(0,1/2.0**.5,[2,NH])
B = numpy.random.normal(0,1,[NH])
V = numpy.random.normal(0,1/NH**.5,[NH])

def forward(X1,X2):
    X = numpy.array([X1.flatten(),X2.flatten()]).T # convert meshgrid into dataset
    Z = X.dot(W)+B
    A = numpy.maximum(0,Z)
    Y = A.dot(V)
    return Y.reshape(X1.shape) # reshape output into meshgrid

# implement backpropagation
def backprop(X1,X2,T):
    X = numpy.array([X1.flatten(),X2.flatten()]).T

    # compute activations
    Z = X.dot(W)+B
    A = numpy.maximum(0,Z)
    Y = A.dot(V)

    # compute backward pass
    DY = (-Y*T>0)*(-T)
    DZ = numpy.outer(DY,V)*(Z>0)

    # compute parameter gradients (averaged over the whole dataset)
    DW = X.T.dot(DZ)
    DB = numpy.mean(DZ,axis=0)
    DV = numpy.mean(numpy.outer(DY,numpy.mean(A,axis=0)),axis=0)

    return DW,DB,DV

# training with gradient descent and visualizing
eta = 0.1
for i in range(128):
    if i in [0,1,3,7,15,31,63,127]:
        Yg = forward(X1g,X2g)
        Yd = forward(X1d,X2d)
        Ed = numpy.maximum(0,-Yd*Td).mean()
        plot(Yg,title="It: %d, Error: %.3f"%(i,Ed))

    DW,DB,DV = backprop(X1d,X2d,Td)
    W = W - DW*eta
    B = B - DB*eta
    V = V - DV*eta
