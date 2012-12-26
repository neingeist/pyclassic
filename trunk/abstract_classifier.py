#!/usr/bin/env python
#coding=utf-8

import numpy

def measure_accuracy(Y, o, threshold=0):
  oo = o.copy()
  oo[numpy.where(o>threshold)[0]] = 1
  oo[numpy.where(o<threshold)[0]] = -1
  d = oo - Y
  return len(d[numpy.where(d==0)[0]])*1.0/len(Y)

class Classifier(object):
  def __init__(self):
    self.verbose = False
    pass

  def set_training_sample(self,X,Y,w=None):
    self.X = X
    self.Y = Y
    if w:
      self.weights = w
    else:
      self.set_uniform_weights()

  def set_uniform_weights(self):
    """ Set all examples to have equal uniform weights. """
    N = len(self.Y)
    weights = (1.0/N)*numpy.ones(N)
    self.weights = weights
