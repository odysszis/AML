import timeit

import numpy
import pickle
import theano
import theano.tensor as T

from logisticRegGPU import LogisticRegression, load_data
from LeNetGPU import LeNetConvPoolLayer

def pre_training(learning_rate = 0.1, n_epochs = 1000, nkerns = 100, batch_size = 260, CNN_inputFilters_path = None, CNN_inputBias_path = None):

    ######################
    #   INITIALIZATIONS  #
    ######################

    # load Auto-encoder pre-trained bias
    if CNN_inputBias_path is None:
        b_CNN_input = None
    else:
        b_CNN_input = theano.shared(
            value=numpy.load(CNN_inputBias_path),       # b is 100 x 1, is ok
            name='b_CNN_input',
            borrow = True
        )

    # load Auto-encoder pre-trained filter weights
    if CNN_inputFilters_path is None:
        W_CNN_input = None
    else:
        W = numpy.load(CNN_inputFilters_path)           # 100 x 11 x 11
        W = numpy.reshape(W, (nkerns, 11*11))           # 100 x 121
        W = W.transpose()                               # 121 x 100
        W = numpy.reshape(W, (1,11,11,batch_size))      # 1 x 11 x 11 x 100
        W_CNN_input = theano.shared(
            value=W,
            name='W_CNN_input',
            borrow = True
        )

    # initialize random generator
    rng = numpy.random.RandomState(23455)

    # load data set
    datasets = load_data()
    train_set_x, train_set_y = datasets[0]
    # compute number of mini-batches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size                                           # 130

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 2D mask vector

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Convolution + Pooling Layer

    # reshape input: train_set_x is batch_size x 4096 and we need it to be 1 x 64 x 64 x batch_size
    layer0_input = x.transpose()
    layer0_input = layer0_input.reshape((1,64,64,batch_size))
    layer0 = LeNetConvPoolLayer(
        rng = rng,
        input=layer0_input,
        filter_shape=(1, 11, 11, nkerns),
        image_shape=(1, 64, 64, batch_size),
        poolsize=(6, 6),
        W = W_CNN_input,
        b = b_CNN_input
    )
    layer0_output = layer0.output.flatten(2)                # batch_size x 8,100

    # Logistic Regression Layer
    layer3 = LogisticRegression(input = layer0_output, n_in = 8100, n_out = 1024)
    layer3_output = layer3.output                           # batch_size x 1024 tensor

    # cost for pre training
    #cost = 0.5 / batch_size * T.sum( ( layer3_output - y )**2 )
    #cost += 0.0001 / 2 * ( T.sum( T.sum( layer3.params[0] ** 2 ) ) )
    # for now ignore regularization
    cost = T.mean((layer3_output - y) ** 2)

    # parameters to be updated
    params = layer3.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    epoch = 0

    while (epoch < n_epochs):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):

            cost_ij = train_model(minibatch_index)
            print '\nepoch = %s' % epoch
            print 'batch = %s' % minibatch_index
            print 'cost = %s' % cost_ij


    print('Optimization complete.')

    with open('../data/logistic_params.pickle', 'w') as f:
        pickle.dump([params], f)


if __name__ == '__main__':
    # use batch_size = 260 in order to do training with only one batch
    # that is train with whole dataset in each epoch
    pre_training(n_epochs=150, batch_size=260,
                 CNN_inputFilters_path='../data/CNN_inputFilters',
                 CNN_inputBias_path='../data/CNN_inputBias')          # use batch_size = 260 to pass all images in each epoch
