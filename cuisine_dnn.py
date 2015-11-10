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

from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from DataInterface import DataInterface
from time import time

import theano
import gzip
import cPickle as pkl

# from pylab import *
import numpy as np
import matplotlib.pyplot as plt

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

# Convert to theano types
x_train = x_train.astype(dtype=floatX)
x_valid = x_valid.astype(dtype=floatX)
x_test  = x_test.astype(dtype=floatX)

y_train = y_train.astype(dtype=np.int32)
y_valid = y_valid.astype(dtype=np.int32)

#==========================#
##  NETWORK ARCHITECTURE  ##
#==========================#
layers=[
        (InputLayer,     {'shape': (None,NUM_FEATURES)}),
        (DenseLayer,     {'num_units':  850, 'nonlinearity':relu}),
        (DropoutLayer,   {'p':0.5}),
        (DenseLayer,     {'num_units':  500, 'nonlinearity':relu}),
        (DropoutLayer,   {'p':0.5}),
        (DenseLayer,     {'num_units':  256}),                     
        (DenseLayer,     {'num_units':NUM_CLASSES, 'nonlinearity':softmax}),
    ]

net = NeuralNet(
        layers=layers,
        max_epochs=150,
        update=nesterov_momentum,
        update_learning_rate=0.001,
        update_momentum=0.9,
        train_split=TrainSplit(eval_size=0.25),
        verbose=1,
    )

#======================#
##  NETWORK TRAINING  ##
#======================#
start = time()
net.fit(x_train,y_train)
end   = time()

m, s = divmod(end-start, 60)
h, m = divmod(m, 60)
print('Training runtime: {0}hrs, {1}mins, {2}s'.format(h,m,s))

## Save network
netfile='./cuisine_shallow_net.pkl.gz'
with gzip.open(netfile, 'wb') as file:
    pkl.dump(net, file, -1)

## Load network
# with gzip.open(netfile, 'rb') as f:
#     net_pretrain = pkl.load(f)



#================#
##  VALIDATION  ##
#================#
y_pred = net.predict(x_valid)

acc = accuracy_score(y_valid,y_pred)
print('Total Accuracy: {0:2.4}%'.format(acc*100))
# cm = confusion_matrix(y_valid, y_pred)
# cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print(cm_norm)


## Fit to validation 
# print('fitting to validation data ...')
# net.fit(x_valid,y_valid,epochs=50)
# with gzip.open('./cuisine_net2.pkl', 'wb') as file:
#     pkl.dump(net, file, -1)

#==========================#
##  TESTING & SUBMISSION  ##
#==========================#
predictions = net.predict(x_test)
dface.make_submission(predictions, filename='dnn_shallow_submission.csv')


