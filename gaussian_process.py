from scipy.linalg import norm
import numpy as np

class GP_Regressor:

    def __init__(self,Xtrain,Ytrain,width,noise): # initialize the parameters

        self.Xtrain=Xtrain
        self.Ytrain=Ytrain
        self.width=width
        self.noise=noise
        self.I=np.identity(len(self.Xtrain)) # identity matrix
        self.Sigma=utils.gaussianKernel(self.Xtrain,self.Xtrain,self.width)+(noise**2)*self.I
        self.Sigma_inv=np.linalg.inv(self.Sigma)


    def predict(self,Xtest):
        self.Xtest = Xtest
        self.I=np.identity(len(self.Xtest)) # identity matrix

        self.Sigma_star=utils.gaussianKernel(self.Xtrain,self.Xtest,self.width)
        self.Sigma_star_trans=self.Sigma_star.T
        self.Sigma_starstar=utils.gaussianKernel(self.Xtest,self.Xtest,self.width)+(noise**2)*self.I
        mean=self.Sigma_star_trans.dot(np.linalg.inv(self.Sigma).dot(self.Ytrain)) # mean
        C=self.Sigma_starstar-self.Sigma_star_trans.dot(np.linalg.inv(self.Sigma).dot(self.Sigma_star)) # covariance
        return mean,C

    def loglikelihood(self,Xtest,Ytest):
        mean,C=self.predict(Xtest)
        C_inv=np.linalg.inv(C)
        logp_yf=-0.5*((Ytest-mean).T).dot(C_inv.dot((Ytest-mean)))-0.5*np.log(np.linalg.det(C_inv))-(len(Xtest)/2)*np.log(2*np.pi)
        return logp_yf
