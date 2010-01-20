#!/usr/bin/env python
#coding=utf-8

import unittest
import numpy
from adaboost import AdaBoost
from decision_stump import DecisionStump


def load_bupa_dataset():
    """
    The BUPA dataset can be obtained from
    http://www.cs.huji.ac.il/~shais/datasets/ClassificationDatasets.html
    See description of this dataset at
    http://www.cs.huji.ac.il/~shais/datasets/bupa/bupa.names
    """
    data = numpy.loadtxt('bupa.data',delimiter = ',')
    X = data[:,:-1] # features
    X = X.T
    Y = data[:,-1]
    Y[Y==2] = -1    # labels <- {1, -1}
    return X, Y


class AdaBoostTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testBupaData(self):
        X, Y = load_bupa_dataset()
        T = 200
        classifier = AdaBoost(DecisionStump)
        accuracy, o, Y  = classifier.test_on_training_set(X,Y,T)
        print accuracy
        self.failUnless(accuracy > .8)

def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(AdaBoostTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    main()




"""

X = loadtxt('trn_X.txt')
y = loadtxt('trn_y.txt',dtype=type(1))

tX = loadtxt('tst_X.txt')
ty = loadtxt('tst_y.txt',dtype=type(1))

td = rank1_metric.TrainingData(X,y)
classifier = adaboost.AdaBoost(rank1_metric.Rank1_Metric)
o = classifier.test_on_training_set(td.X,td.Y,200)

td2 = rank1_metric.TrainingData(tX,ty)
classifier2 = adaboost.AdaBoost(rank1_metric.Rank1_Metric)
o = classifier2.test_on_training_set(td2.X,td2.Y,200)


threshold = 0
oo = o.copy()
oo[numpy.where(o>threshold)[0]] = 1
oo[numpy.where(o<threshold)[0]] = -1

cascading

 selected balanced positive and negative samples

P -> N: missing
N -> P: imposer

assymetric

  level = 0
  while (True):
     sample (with no replacement) from training pool
     train adaboost
     verify on training set
     tune the threshold s.t. the missing rate is very low
     if FP/FN < ratio: break
     level += 1
     keep FP, FN instances









"""
