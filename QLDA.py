import math
import numpy

"""
import QLDA
import numpy
training_X = numpy.random.rand(1000,5)
training_Y = numpy.random.rand(1000,1)
training_Y[training_Y>.5]=1
training_Y[training_Y<.5]=-1
qlda_classifier = QLDA.QLDAClassifier()
qlda_classifier.Train(training_X, training_Y)
test_X = training_X.copy()
[test_Y, test_Y_raw] = qlda_classifier.Classify(test_X)
"""

def compute_mean_covariance(X):
    d = X.shape[1]
    m = numpy.mean(X,0)
    C = numpy.cov(X,rowvar=0)
    [ew,ev] = numpy.linalg.eig(C)
    eps = 1e-10
    if ew.min()<eps: # in case C1 is positive semidefinite
        ew_max = ew.max()
        C = C + ew_max*numpy.eye(d) # make C1 positive definite
        ew = ew + ew_max
    c = 1/math.sqrt( math.pow((2*math.pi),d) * ew.prod() );
    invC = numpy.dot(numpy.dot(ev,numpy.diag(1/ew)),ev.T);
    return m,c,invC

class QLDAClassifier:
    def __init__(self):
        self.parameter = .5
        self.threshold = 0
        
    def Train(self,X,Y):
        """ Train a QLDA classifier """
        X1 = X[numpy.where(Y==1)[0]]
        X2 = X[numpy.where(Y==-1)[0]]
        m1,c1,invC1 = compute_mean_covariance(X1)
        m2,c2,invC2 = compute_mean_covariance(X2)
        self.m1 = m1
        self.c1 = c1
        self.invC1 = invC1
        self.m2 = m2
        self.c2 = c2
        self.invC2 = invC2

    def Classify(self,X):
        # Get the likelihood for both classes
        n = X.shape[0]
        O1 = X - numpy.ones([n,1])*self.m1
        O2 = X - numpy.ones([n,1])*self.m2
        f1 = self.c1*numpy.exp(-0.5*(numpy.dot(O1,self.invC1)*O1).sum(1))
        f2 = self.c2*numpy.exp(-0.5*(numpy.dot(O2,self.invC2)*O2).sum(1))

        # Normalize the values
        F = f1 + f2
        F[F==0] = 1
        f1 = f1/F
        f2 = f2/F
        Y_raw = self.parameter * f1 - (1-self.parameter) * f2;
        Y = Y_raw.copy()
        Y[Y_raw>=self.threshold]=1
        Y[Y_raw<self.threshold]=-1
        return Y,Y_raw
    
    


