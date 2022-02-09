import numpy as np
import scipy.integrate

def pdf(X,THETA):
  return (1.0 / np.pi) * (1.0 / (1+(X-THETA)**2))

def ll(D,THETA):
  pr = np.array([pdf(x,THETA) for x in D]) pr = np.log(pr)
  loglik = np.sum(pr,0)
  return loglik

class MLClassifier:
    def fit(self,THETA,D1,D2):
        self.theta1=THETA[np.argmax(ll(D1,THETA))]
        self.theta2=THETA[np.argmax(ll(D2,THETA))]
        return self.theta1,self.theta2
    def predict(self,X,p1,p2):
        g = np.log(pdf(X,self.theta1))-np.log(pdf(X,self.theta2))+np.log(p1)-np.log(p2)
        return g

def prior(THETA):
    return (1/(10*np.pi))*(1/(1+(THETA/10)**2))
def posterior(D,THETA):
    likelihood = np.array([pdf(x,THETA) for x in D])
    likelihood = np.sum(likelihood,0)
    num = likelihood*prior(THETA)
    den = scipy.integrate.trapz(num)
    post = num/den
    return post

class BayesClassifier:
    def fit(self,THETA,D1,D2):
        self.post1 = posterior(D1,THETA)
        self.post2 = posterior(D2,THETA)
    def predict(self,X,p1,p2):
        pd1 = [pdf(x,THETA) for x in X] * self.post1
        pd2 = [pdf(x,THETA) for x in X] * self.post2
        h = np.log(scipy.integrate.trapz(pd1))-np.log(scipy.integrate.trapz(pd2))
        return h
