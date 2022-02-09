import numpy as np,scipy, scipy.spatial
import sklearn, sklearn.datasets

def getGaussianKernel(X1,X2,scale):
    K = np.exp(-scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')/
               (2*scale**2))
    return K

# matrices for the CVXOPT solver
import cvxopt,cvxopt.solvers
cvxopt.solvers.options['show_progress'] = False

def getQPMatrices(K,T,C):
    N = len(K)
    P = cvxopt.matrix(np.outer(T,T)*K)
    q = cvxopt.matrix(-np.ones(N))
    G = cvxopt.matrix(np.concatenate((-np.eye(N),np.eye(N))))
    h = cvxopt.matrix(np.concatenate((np.zeros(N),np.ones(N)*C)))
    A = cvxopt.matrix(T,(1,N))
    b = cvxopt.matrix([0.0])
    return P,q,G,h,A,b

# compute bias parameter theta
def getTheta(K,T,alpha,C):
    # find support vectors
    idx = np.argmin(np.abs(alpha-C/2.0))
    theta = np.sum(T[idx]-np.dot(K[idx,:],alpha*T))
    return theta

class GaussianSVM:

    def __init__(self,C=1.0,scale=1.0):
        self.C, self.scale = C, scale

    def fit(self,X,T):
        K = getGaussianKernel(X,X,self.scale)
        P,q,G,h,A,b = getQPMatrices(K,T,self.C)
        sol = cvxopt.solvers.qp(P,q,G,h,A,b)
        alpha = np.array(sol['x']).flatten()
        self.theta = getTheta(K,T,alpha,self.C)
        idx = np.abs(alpha)>0
        self.alpha = alpha[idx]
        self.X = X[idx]
        self.T = T[idx]

    def predict(self,X):
        K = getGaussianKernel(X,self.X,self.scale)
        Y = np.sign(np.dot(K,self.alpha*self.T)+self.theta)
        return Y

# training and testing
D = sklearn.datasets.load_breast_cancer()
X = D['data']
T = D['target']
T = (D['target']==1)*2.0-1.0

for scale in [30,100,300,1000,3000]:
    for C in [10,100,1000,10000]:

        acctrain,acctest,nbsvs = [],[],[]

        svm = GaussianSVM(C=C,scale=scale)

        for i in range(10):

            # Split the data
            R = np.random.mtrand.RandomState(i).permutation(len(X))
            Xtrain,Xtest = X[R[:len(R)//2]]*1,X[R[len(R)//2:]]*1
            Ttrain,Ttest = T[R[:len(R)//2]]*1,T[R[len(R)//2:]]*1

            # Train and test the SVM
            svm.fit(Xtrain,Ttrain)
            acctrain += [(svm.predict(Xtrain)==Ttrain).mean()]
            acctest  += [(svm.predict(Xtest)==Ttest).mean()]
            nbsvs += [len(svm.X)*1.0]

        print('scale=%9.1f  C=%9.1f  nSV: %4d  train: %.3f  test: %.3f'%(
            scale,C,np.mean(nbsvs),np.mean(acctrain),np.mean(acctest)))
    print('')
