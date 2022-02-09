import numpy as np,
import sklearn,sklearn.datasets,utils
import time

D = sklearn.datasets.fetch_lfw_people(resize=0.5)['images']
D = D[np.random.mtrand.RandomState(1).permutation(len(D))[:2000]]*1.0
D = D - D.mean(axis=(1,2),keepdims=True)
D = D / D.std(axis=(1,2),keepdims=True)
print(D.shape)

utils.scatterplot(D[:,32,20],D[:,32,21]) # plot relation between adjacent pixels
utils.render(D[:30],15,2,vmax=5)         # display first 10 examples in the data

# PCA with SVD
N = len(D)
X = D.reshape(len(D), -1).T
X -= X.mean(axis=1,keepdims=True)
Z = X/np.sqrt(N)
U,L,V = np.linalg.svd(Z, full_matrices = False)

# plot the projection of the dataset on the first two principle components
pc1 = U[:,0].dot(X)
pc2 = U[:,1].dot(X)
utils.scatterplot(pc1,pc2)

# PCA with power iteration
S = (1/N)*(X.dot(X.T))
w = np.random.random(len(X))
jw = []
j_prev = float('nan')
for i in range(100):
    Sw = S.dot(w)
    j = w.dot(Sw)
    print('iter: %2d, j(w) val: %10.5f' %(i, j))
    if j-j_prev<0.01:
        print('done')
        break
    w = sw/np.linalg.norm(sw)
    j_prev = j

# visualize the eigenvector
utils.render(w,1,1)
