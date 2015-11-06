# Description: Python script to train deep neural network on whatscooking project data
# Dependencies: DataInterface, 

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer

from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import rectify as relu
from lasagne.updates import adam, nesterov_momentum
from lasagne.layers import get_all_params

## Nolearn Modules
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from DataInterface import DataInterface
from pylab import *

import theano
import cPickle as pkl

floatX = theano.config.floatX


#==================#
##  LOADING DATA  ##
#==================#
dface  = DataInterface()
x_train,x_valid,y_train,y_valid = dface.get_traindata(full=True)
x_test = dface.get_testdata()
labels = dface.classes

NUM_FEATURES = x_train.shape[1]
NUM_CLASSES  = len(labels)


#==========================#
##  NETWORK ARCHITECTURE  ##
#==========================#
layers=[
        (InputLayer,     {'shape': (None,1,NUM_FEATURES)}),
        (DenseLayer,     {'num_units': 1000, 'nonlinearity':relu}),
        (DropoutLayer,   {'p':0.5}),
        (DenseLayer,     {'num_units':  500, 'nonlinearity':relu}),
        (DropoutLayer,   {'p':0.5}),
        (DenseLayer,     {'num_units':  500, 'nonlinearity':relu}),
        (DropoutLayer,   {'p':0.5}),
        (DenseLayer,     {'num_units':  500}),
        (DropoutLayer,   {'p':0.5}),
        (DenseLayer,     {'num_units':  256}),
        (DropoutLayer,   {'p':0.5}),                         
        (DenseLayer,     {'num_units':NUM_CLASSES, 'nonlinearity':softmax}),
    ]

net = NeuralNet(
        layers=layer1,
        max_epochs=40,
        update=nesterov_momentum,
        update_learning_rate=0.001,
        update_momentum=0.9,
        train_split=TrainSplit(eval_size=0.25),
        verbose=1,
    )

#======================#
##  NETWORK TRAINING  ##
#======================#
net.fit(X_train,y_train)
