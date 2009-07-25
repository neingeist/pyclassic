#!/usr/bin/env python
#coding=utf-8

import unittest
import numpy
import adaboost


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
        accuracy, o, Y  = adaboost.test_on_training(X,Y,T)
        print accuracy
        self.failUnless(accuracy > .8)

def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(AdaBoostTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    main()





