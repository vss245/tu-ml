import numpy as np

def getdata(seed):

  n = 10              # data points
  d = 50              # dimensionality of data
  m = np.ones([d]) # true mean
  s = 1.0             # true standard deviation

  rstate = np.random.mtrand.RandomState(seed)
  X = rstate.normal(0,1,[n,d])*s+m

  return X,m,s

# maximum likelihood estimator from a sample of the data assuming Gaussian
def ML(X):
  return X.mean(axis=0)

# James-Stein estimator
def JS(X,s):
    d = X.shape[1]
    N = X.shape[0]
    m_JS = (1-((d-2)*(s/N))/((np.linalg.norm(ML(X)))**2))*ML(X)
    return m_JS

# KL-divergence based bias-variance decomposition - regression
def biasVarianceRegression(sampler, predictor, X, T, nbsamples):
    out = []
    for i in range(nbsamples):
        ds = sampler.sample() # this outputs a sample from X and corresponding T
        pr = predictor.fit(ds[0],ds[1])
        out.append(pr.predict(X))
    out = np.array(out)
    bias = np.linalg.norm(ML(out-T))**2
    diff = out - ML(out)
    variance = np.mean(np.matmul(diff,diff.T).diagonal())
    return bias,variance

# KL-divergence based bias-variance decomposition - classification
def KL(a,b,axis):
    return np.sum(a*np.log(a/b),axis=axis)

def biasVarianceClassification(sampler, predictor, X, T, nbsamples=25):
    out = []
    for i in range(nbsamples):
        ds = sampler.sample()
        pr = predictor.fit(ds[0],ds[1])
        out.append(pr.predict(X))
    # calculate R
    num = np.exp(np.mean(np.log(out),axis=0))
    R = num/num.sum(axis=1,keepdims=True)
    bias = np.mean(KL(T,R,1))
    variance = np.mean(KL(R,out,2))
    return bias,variance
