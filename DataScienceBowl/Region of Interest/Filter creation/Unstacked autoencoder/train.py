import theano.tensor as T
import numpy as np
import timeit
from theano.tensor.shared_randomstreams import RandomStreams
import theano
from autoencoder import dA
import sys


def train_model(train_data, n_epochs, model_class, **args):

    """
    trains auto-encoder model for initialising weights for the CNN layer of the model, taking as input
    random mini batches from the training images.
    :param training data
    :param n_epochs: number of training iterations
    :param model_class: class of model to train
    :param **costparams: any named inputs required by the cost function


    RETURNS: final array of weights from trained model
    """

    traindim = train_data.shape

    X = T.matrix('X')
    Xbatch = T.matrix('Xbatch')

    train_data = theano.shared(train_data)

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    model_object = model_class(numpy_rng=rng,
            theano_rng=theano_rng,
            input=X,
            input_batch=Xbatch,
            n_visible=121,
            n_hidden=100)

    cost, updates = model_object.get_cost_updates(batchdim=traindim, **args)

    train_model = theano.function(inputs=[], outputs=cost, updates=updates,
                                   givens={Xbatch: train_data})
    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(n_epochs):
        c = train_model() #compute cost
        print 'Training epoch %d, cost ' % epoch, c
        end_time = timeit.default_timer()
        training_time = (end_time - start_time)

    params = model_object.Whid.get_value()

    return params



if __name__ == "__main__":

    # load sunny data and collapse to correct dim

    train = np.load('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SBtrainImage')
    train_batch = np.load('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SBtrainImage_batch')
    dim = train.shape
    train = np.reshape(train, (dim[0], (dim[1]*dim[2])))
    dim = train_batch.shape
    train_batch = np.reshape(train_batch, (dim[0], (dim[1]*dim[2])))
    train_batch  = np.array(train_batch, dtype='float64')

    params_final = train_model(train_data=train_batch, n_epochs = 10000, model_class = dA, learning_rate=10, lam=10^4)


# TODO extract and store output from dA trained instance for reuse in subsequent steps. 11*11*100
# TODO include regularisation terms with tensor operations
# TODO wrap train in function, maybe pass **args
# TODO Add parameter for number of hidden layers into autoencoder
# TODO stacked autoencoder
