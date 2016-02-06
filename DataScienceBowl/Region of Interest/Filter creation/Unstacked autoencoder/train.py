import theano.tensor as T
import numpy as np
import timeit
from theano.tensor.shared_randomstreams import RandomStreams
import theano
from autoencoder import dA
import sys


def train_model(train_data, numbatches, n_epochs, model_class, **args):

    """
    trains auto-encoder model for initialising weights for the CNN layer of the model, taking as input
    random mini batches from the training images.
    :param training data
    :param n_epochs: number of training iterations
    :param model_class: class of model to train
    :param **args: any named inputs required by the cost function


    RETURNS: final array of weights from trained model
    """

    traindim = train_data.shape
    batch_size = traindim[0]/numbatches

    X = T.matrix('X')
    index = T.lscalar()

    train_data = theano.shared(train_data)
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    model_object = model_class(numpy_rng=rng,
            theano_rng=theano_rng,
            input=X,
            n_visible=121,
            n_hidden=100)

    cost, updates = model_object.get_cost_updates( **args)

    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                                   givens={X: train_data[index * batch_size:(index + 1) * batch_size]})


    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(n_epochs):
        for nindex in range(numbatches):
            c = train_model(nindex) #compute cost
            print 'Training epoch %d, batchId, cost' % epoch, nindex, c

    params = model_object.Whid.get_value()

    return params

if __name__ == "__main__":

    # load sunny data and collapse to correct dim

    train = np.load('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SBtrainImage_batch')
    dim = train.shape
    train = np.reshape(train, (dim[0], (dim[1]*dim[2])))
    train = np.array(train, dtype='float64')
    numbatches = 5
    batchdim = train[0]/5

    params_final = train_model(train_data=train, numbatches = numbatches, n_epochs = 10000,
                               model_class = dA, datadim = batchdim, learning_rate=10, lam=10^4)


# TODO extract and store output from dA trained instance for reuse in subsequent steps. 11*11*100
# TODO include regularisation terms with tensor operations
# TODO wrap train in function, maybe pass **args
# TODO Add parameter for number of hidden layers into autoencoder
# TODO stacked autoencoder
