import utils, numpy as np

def sw(X):
    return ((X-X.mean(0)).T).dot(X-X.mean(0))

def fisher(X1,X2): # return Fisher linear discriminant
    dm = X2.mean(0)-X1.mean(0)
    sw_all = sw(X1)+sw(X2)
    return np.linalg.inv(sw_all).dot(dm)

def objective(X1,X2,w): # evaluate the objective for an arbitrary projection vector w
    dm = X2.mean(0)-X1.mean(0)
    wsbw = (w.dot(dm))**2
    wsww = (w.dot(sw(X1)+sw(X2))).dot(w)
    return wsbw/wsww

def expand(X): # quadratic expansion
    return np.array([np.concatenate([x,np.outer(x,x)[np.triu_indices(X.shape[1])]]) for x in X])
