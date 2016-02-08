"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import pickle as cPickle
import gzip
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (batch_size, n_in, n_out)
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.output = T.tanh( T.dot(input, self.W) + self.b )   # 20 x 1024

        # parameters of the model
        self.params = [self.W, self.b]                                      # W: 1024 x 8100, b: 1024 x 1

        # keep track of model input
        self.input = input


def load_data():
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############


    print('... loading data')

    train_set_x = np.load('../data/SBtrainImage')
    train_set_y = np.load('../data/SBtrainMask32')

    print train_set_x[1,10:20,10:20]
    print train_set_y[1,10:20,10:20]

    train_set_x = np.asarray(train_set_x, dtype='float64')
    dim = train_set_x.shape
    train_set_x = np.reshape(train_set_x, (dim[0], (dim[1]*dim[2])) )
    shared_x = theano.shared(train_set_x, borrow=True)                      # convert to 260 x 4096

    train_set_y = np.asarray(train_set_y, dtype='float64')
    dim = train_set_y.shape
    train_set_y = np.reshape(train_set_y, (dim[0], (dim[1]*dim[2])) )
    shared_y = theano.shared(train_set_y, borrow=True)                      # convert to 260 x 1024
    #shared_y = T.cast(shared_y, 'int64')

    rval = [(shared_x, shared_y)]

    return rval


if __name__ == '__main__':
    load_data()
    #with open('preTrainLogReg.pickle') as f:
    #    params = pickle.load(f)

    #weights, b = params[0]
    #print weights.get_value(borrow=True).shape
    #print b.get_value(borrow=True).shape
