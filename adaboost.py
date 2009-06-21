import math
import numpy
import operator

class Stump:
    """1D stump"""
    def __init__(self, err, threshold, s):
        self.err = err
        self.threshold = threshold
        self.s = s

    def __cmp__(self, other):
        return cmp(self.err, other.err)

class Classifier:
    def __init__(self):
        pass

    def SetTrainingSample(self,X,Y):
        self.X = X
        self.Y = Y

    def SetWeights(self, weights):
        # todo:
        #    d,N = self.X.shape
        #    #Initialize the weights to a uniform distribution
        #    weights = (1.0/N)*numpy.ones(N)
        self.weights = weights

class DecisionStump(Classifier):
    def __init__(self):
        Classifier.__init__(self)

    def Train(self):
        X = self.X
        Y = self.Y
        w = self.weights

        feature_index, threshold = train_decision_stump(X,Y,w)
        self.feature_index = feature_index
        self.threshold = threshold

    def Predict(self,X):
        d,N = X.shape
        feature_index = self.feature_index
        threshold = self.threshold

        Y = numpy.ones(N)
        Y[numpy.where(X[feature_index]<threshold)[0]] = -1
        return Y


class AdaBoost(Classifier):
    def __init__(self):
        Classifier.__init__(self)
        self.verbose = False
        self.WeakClassifier = DecisionStump

    def SetWeakClassifier(self, weakClassifier):
        self.WeakClassifier = weakClassifier

    def Train(self, T):
        X = self.X
        Y = self.Y
        d,N = X.shape
        w = (1.0/N)*numpy.ones(N)
        self.WeakClassifiers = []
        self.alpha = []
        for t in range(T):
            weakLearner = self.WeakClassifier()
            weakLearner.SetTrainingSample(X,Y)
            weakLearner.SetWeights(w)
            weakLearner.Train()
            Ypred = weakLearner.Predict(X)
            e = (sum(0.5*w*abs((Y-Ypred)))/sum(w))
            alpha = 0.5*math.log((1-e)/e)
            w *= numpy.exp(-alpha*Y*Ypred)
            w /= sum(w)
            self.WeakClassifiers.append(weakLearner)
            self.alpha.append(alpha)
        self.T = T

    def Predict(self,X):
        d,N = X.shape
        Y = numpy.zeros(N)
        for t in range(self.T):
            weakLearner = self.WeakClassifiers[t]
            Y += self.alpha[t]*weakLearner.Predict(X)
        return Y



def train_decision_stump(X,Y,w):
    stumps = [build_stump_1d(x,Y,w) for x in X]
    feature_index, best_stump = min(enumerate(stumps), key=operator.itemgetter(1))
    best_threshold = best_stump.threshold
    return feature_index, best_threshold


def build_stump_1d(x,y,w):
    sorted_xyw = numpy.array(sorted(zip(x,y,w), key=operator.itemgetter(0)))
    xsorted = sorted_xyw[:,0]
    wy = sorted_xyw[:,1]*sorted_xyw[:,2]
    score_left = numpy.cumsum(wy)
    score_right = numpy.cumsum(wy[::-1])
    score = -score_left[0:-1:1] + score_right[-1:0:-1]
    Idec = numpy.where(xsorted[:-1]<xsorted[1:])[0]
    if len(Idec)>0:  # determine the boundary
        ind, maxscore = max(zip(Idec,abs(score[Idec])),key=operator.itemgetter(1))
        err = 0.5-0.5*maxscore # compute weighted error
        threshold = (xsorted[ind] + xsorted[ind+1])/2 # threshold
        s = numpy.sign(score[ind]) # direction of -1 -> 1 change
    else:  # all identical; todo: add random noise?
        err = 0.5
        threshold = 0
        s = 1
    return Stump(err, threshold, s)


def test_on_training(X,Y,T, threshold=0):
    adaboost = AdaBoost()
    adaboost.SetTrainingSample(X,Y)
    adaboost.Train(T)
    o = adaboost.Predict(X)
    o[numpy.where(o>threshold)[0]] = 1
    o[numpy.where(o<threshold)[0]] = -1
    oY = o*Y
    accuracy = (sum(oY)+len(Y))/(2*len(Y))
    return accuracy, o, Y

def test(T, threshold = 0):
    X = numpy.loadtxt('X.txt')
    Y = numpy.loadtxt('Y.txt')
    return test_on_training(X,Y,T, threshold)


