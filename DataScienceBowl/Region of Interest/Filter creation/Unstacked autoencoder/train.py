import theano.tensor as T
import numpy as np
import timeit
from theano.tensor.shared_randomstreams import RandomStreams
import theano
from autoencoder import dA
import sys


#load data and collapse to correct dim
#inputs to wrap in train function

train = np.load('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SBtrainImage')
train_batch = np.load('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SBtrainImage_batch')
dim = train.shape
train = np.reshape(train, (dim[0], (dim[1]*dim[2])))
dim = train_batch.shape
train_batch = np.reshape(train_batch, (dim[0], (dim[1]*dim[2])))
train_batch  = np.array(train_batch, dtype='float64')
batchdim = train_batch.shape
n_epochs=1000
lam = 10^4
lrate = 10

X = T.matrix('X')
Xbatch = T.matrix('Xbatch')

train = theano.shared(train)
train_batch = theano.shared(train_batch)


rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

da = dA(numpy_rng=rng,
        theano_rng=theano_rng,
        input=X,
        input_batch=Xbatch,
        n_visible=121,
        n_hidden=100)

cost, updates = da.get_cost_updates(learning_rate=lrate, lam=lam, batchdim=batchdim)

train_da = theano.function(inputs=[], outputs=cost,
                           updates=updates,
                           givens={Xbatch: train_batch})

start_time = timeit.default_timer()

############
# TRAINING #
############

# go through training epochs
for epoch in xrange(n_epochs):
    c = train_da() #compute cost
    params = da.Whid.get_value()
    print 'Training epoch %d, cost ' % epoch, c
    print params.shape
    end_time = timeit.default_timer()
    training_time = (end_time - start_time)



# TODO extract and store output from dA trained instance for reuse in subsequent steps. 11*11*100
# TODO include regularisation terms with tensor operations
# TODO wrap train in function, maybe pass **args
# TODO Add parameter for number of hidden layers into autoencoder
# TODO stacked autoencoder

