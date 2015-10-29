# Description: Python script to train deep neural network on whatscooking project data
# Dependencies: load_data.py, 

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

