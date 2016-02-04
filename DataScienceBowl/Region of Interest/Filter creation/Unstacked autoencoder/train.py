import theano.tensor as T
import numpy as np
import timeit
from theano.tensor.shared_randomstreams import RandomStreams
import theano
from autoencoder import dA
import sys


#load data and collapse to correct dim
#inputs to wrap in train function

train = np.load('/DataScienceBowl/data/SBtrainImage')
train_batch = np.load('/DataScienceBowl/data/SBtrainImage_batch')
dim = train.shape
train = np.reshape(train, (dim[0], (dim[1]*dim[2])))
dim = train_batch.shape
train_batch = np.reshape(train_batch, (dim[0], (dim[1]*dim[2])))
print(train_batch.shape)
n_epochs=10
lam = 10^4
lrate = 10




x = T.matrix('x')
x_batch = T.matrix('x_batch')


rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

da = dA(numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        input_batch=x_batch,
        n_visible=121,
        n_hidden=100)

cost, updates = da.get_cost_updates(learning_rate=lrate, lam=lam)

train_da = theano.function(outputs=cost, updates=updates,
                           givens={x_batch: train_batch,
                                   x: train})

start_time = timeit.default_timer()

############
# TRAINING #
############

# go through training epochs
for epoch in xrange(n_epochs):
    c = train_da #compute cost

    print 'Training epoch %d, cost ' % epoch, c

    end_time = timeit.default_timer()
    training_time = (end_time - start_time)

# TODO extract and store output from dA trained instance for reuse in subsequent steps

