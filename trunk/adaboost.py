#!/usr/bin/env python
#coding=utf-8

import sys, math, random
import numpy
import operator

from abstract_classifier import Classifier

def shuffle(ary):
    a = len(ary)
    b = a-1
    for d in range(b,0,-1):
        e = random.randint(0,d)
        if e == d:
            continue
        ary[d],ary[e] = ary[e],ary[d]
    return ary


class AdaBoost(Classifier):
    def __init__(self, weak_classifier_type):
        Classifier.__init__(self)
        self.WeakClassifierType = weak_classifier_type

    def train(self, T, k = 1):
        X = self.X
        Y = numpy.array(self.Y)
        N = len(self.Y)
        w = (1.0/N)*numpy.ones(N)
        self.weak_classifier_ensemble = []
        self.alpha = []
        for t in range(T):
            sys.stdout.write('.')
            weak_learner = self.WeakClassifierType()
            weak_learner.set_training_sample(X,Y)
            weak_learner.weights = w
            weak_learner.train()
            Y_pred = weak_learner.predict(X)
            # (Y=-1, Y_pred=1) False Positive
            # (Y=1, Y_pred=-1) Missing  should be assigned more weights
            #ww = numpy.log(k)*(numpy.exp( (Y-Y_pred)>1 ) - 1)/(numpy.exp(1)-1) + 1
            e = sum(0.5*w*abs((Y-Y_pred)))/sum(w)
            #e = sum(0.5*w*abs(Y-Y_pred))
            ee = (1-e)/(e*1.0)
            alpha = 0.5*math.log(ee+0.00001)
            w *= numpy.exp(-alpha*Y*Y_pred) #*ww) # increase weights for wrongly classified
            w /= sum(w)
            self.weak_classifier_ensemble.append(weak_learner)
            self.alpha.append(alpha)
        print "\n"
        self.T = T

    def predict(self,X):
        X = numpy.array(X)
        N, d = X.shape
        Y = numpy.zeros(N)
        for t in range(self.T):
            #sys.stdout.write('.')
            weak_learner = self.weak_classifier_ensemble[t]
            Y += self.alpha[t]*weak_learner.predict(X)
        return Y

    def test_on_training_set(self, X, Y, T):
        self.set_training_sample(X,Y)
        self.train(T)
        o = self.predict(X)
        return o

    def measure_accuracy(self, Y, o, threshold=0):
        oo = o.copy()
        oo[numpy.where(o>threshold)[0]] = 1
        oo[numpy.where(o<threshold)[0]] = -1
        Pos = set(numpy.where(Y==1)[0])
        Neg = set(numpy.where(Y==-1)[0])
        pPos = set(numpy.where(oo==1)[0])
        pNeg = set(numpy.where(oo==-1)[0])
        TP = pPos.intersection(Pos)
        FP = pPos.intersection(Neg)
        TN = pNeg.intersection(Neg)
        FN = pNeg.intersection(Pos)
        TPr = len(TP)*1.0/len(Pos)
        FPr = len(FP)*1.0/len(Neg) # or just len(FP)
        return len(TP), len(FP),

    def plot_ROC(self, Y, o):
        # linspace(min(o), max(o), 100)
        perf = [self.measure_accuracy(Y,o,thres) for thres in numpy.linspace(-1,1,10)]
        return numpy.array(perf)

"""
class CascadeClassifier(Classifier):
    def __init__(self, base_classifier_type):
        self.base_classifier_type = base_classifier_type

    def new_classifier(self, *args):
        return self.base_classifier_type(*args)

    def config(self, max_level):
        pass

    def set_training_pool(self, X, Y):
        #self.X = copy.copy(X)
        #self.Y = copy.copy(Y)
        self.training_data = shuffle(zip(X,Y))
        pool = self.training_data
        self.positive_pool = [(x,y) for x,y in pool if y]
        self.negative_pool = [(x,y) for x,y in pool if not y]
        self.pos_start_ind = 0
        self.pos_remain = len(self.positive_pool)
        self.neg_start_ind = 0
        self.neg_remain = len(self.negative_pool)

    def sample_with_no_repl(self, n_pos, n_neg):
        if self.pos_remain > 0:
            pool = self.positive_pool
            ind = self.pos_start_ind
            pos = pool[ind:ind+n_pos]
            self.pos_start_ind += len(pos)
            self.pos_remain -= len(pos)
        else:
            pos = []
        if self.

        self.pos_start_ind
        X = self.positive_pool[0:min(n_pos,n)]

    def train(self, *args):
        pass


"""

"""

Given X,Y





def cascade(weak_classifier_type, training_pool, max_level, T, k):
    level = 0
    classifiers = []
    X,Y = [],[]
    while (True):
        sample (with no replacement) from training pool -> X,Y
        Pos = set(numpy.where(Y==1)[0])
        Neg = set(numpy.where(Y==-1)[0])
        classifier = Adaclassifier_type(weak_classifier_type)
        classifier.set_training_sample(X,Y)
        classifier.train(T, k)
        Y_pred = classifier.predict(X)
        for thres in numpy.linspace(1,-1,10): #todo
            FP, FN = classifier.measure_accuracy(Y,o,thres)
            if len(FN)/len(Pos) < min_FNr:
                break
        classifiers.append(classifier, thres)
        if len(FP)/len(FN) < ratio: break
        level += 1
        if level >= max_level: break
        FP, FN -> X, Y (w_FN > w_FP)
"""



