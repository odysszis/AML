
import os
import sys
import timeit
import logging
import numpy
import pickle
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

from logisticReg import LogisticRegression, load_data

logging.basicConfig(filename='logistic.log', filemode='w', level=logging.INFO)

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(6, 6), W = None, b = None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape           20 x 1 x 64 x 64

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)                  100 x 1 x 11 x 11

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)                     20 x 1 x 64 x 64

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)

        :type W: tensor with size of filter_shape
        :param W: filter weights

        :type b: tensor of length (filter_shape[0],)
        :param b: bias term of each convolution
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # load filter weights

        if W is None:

            # if W are not provided, generated them randomly

            fan_in = numpy.prod(filter_shape[1:])
            #    pooling size each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                       numpy.prod(poolsize))
            # initialize weights with random weights
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            self.W = W

        # load bias
        if b is None:

            # if b are not provided, generate them randomly

            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )
        # conv_out should be 30 x 100 x 54 x 54

        # apply sigmoid before pooling
        conv_out = T.nnet.sigmoid( conv_out + self.b.dimshuffle('x', 0, 'x', 'x') )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True,
            mode='average_exc_pad'
        )
        # pooled_out should be 30 x 100 x 9 x 9

        # no padding to preserve shape
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        # CHANGED
        #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.output = pooled_out

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


def fine_tuning(learning_rate = 0.1, n_epochs = 200, nkerns = 100, batch_size = 20):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: int
    :param nkerns: number of convolution layer filters (kernels)

    :type batch_size: int
    :param batch_size: size of batch in which the data are passed to the model
    """

    ######################
    #   INITIALIZATIONS  #
    ######################

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size                                           # 13

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 64 * 64)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 64, 64))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (64-11+1 , 64-11+1) = (54, 54)
    # maxpooling reduces this further to (54/6, 54/6) = (9, 9)
    # 4D output tensor is thus of shape (batch_size, 100, 9, 9)
    layer0 = LeNetConvPoolLayer(
        rng = rng,
        input = layer0_input,
        filter_shape = (nkerns, 1, 11, 11),
        image_shape = (batch_size, 1, 64, 64),                # 20 x 100 x 11 x 11
        poolsize = (6, 6)
    )
    #rng, input, W, b, filter_shape, image_shape, poolsize=(6, 6)

    layer0_output = layer0.output.flatten(2)                # 20 x 8,100

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input = layer0_output, n_in = 8100, n_out = 1024)

    layer3_output = layer3.output                           # 20 x 1024 tensor

    #cost = 0.5 / batch_size * T.sum( T.sum( layer3_output - y )**2 )    # 20x1024 - 20x1024
    #cost += 0.9 / 2 * ( T.sum( T.sum( layer3.params[0] ** 2 ) ) )
    #cost += 0.9 / 2 * ( T.sum( T.sum( T.sum( T.sum( layer0.params[0] ) ) ) ) )
    cost = T.mean((layer3_output - y) ** 2)

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
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
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

    print('Optimization complete.')



def predict(nkerns = 100, batch_size = 20, logistic_params_path = None, CNN_inputFilters_path = None, CNN_inputBias_path = None):

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
        W = numpy.load(CNN_inputFilters_path)
        W_4D_tensor = numpy.reshape(W, (100,1,11,11))
        W_CNN_input = theano.shared(
            value=W_4D_tensor,    # W is 100 x 11 x 11 should convert to 100 x 1 x 11 x 11
            name='W_CNN_input',
            borrow = True
        )

    # load logistic layer pre-training parameters
    if logistic_params_path is None:
        W_logistic = None
        b_logistic = None
    else:
        with open('logRegPreTrainParams.pickle') as f:
            params = pickle.load(f)
        W_logistic, b_logistic = params[0]
        print type(W_logistic), type(b_logistic)

    rng = numpy.random.RandomState(23455)

    # load data
    datasets = load_data()
    train_set_x, train_set_y = datasets[0]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    ###############
    # BUILD MODEL #
    ###############

    # build model
    print('... building the model')
    index = T.lscalar()
    x = T.matrix('x')
    y = T.matrix('y')

    # Convolution + Pooling Layer
    layer0_input = x.reshape((batch_size, 1, 64, 64))
    layer0 = LeNetConvPoolLayer(                                                    # cnn + pooling
        rng = rng,
        input = layer0_input,
        filter_shape = (nkerns, 1, 11, 11),
        image_shape = (batch_size, 1, 64, 64),
        poolsize = (6, 6),
        W = W_CNN_input,
        b = b_CNN_input
    )
    layer0_output = layer0.output.flatten(2)

    # Logistic Regression Layer
    layer3 = LogisticRegression(                                                    # logistic
        input = layer0_output, n_in = 8100, n_out = 1024,
        W = W_logistic, b = b_logistic
    )
    predict_model = theano.function(
        inputs = [index],
        outputs=layer3.output,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    im_out = predict_model(0)
    print im_out.shape
    im_out = numpy.reshape(im_out, (32,32))

    pylab.imshow(im_out)

if __name__ == '__main__':
    #fine_tuning()
    predict(batch_size=1, logistic_params_path = 'logisticParams_150epochs.pickle',
            CNN_inputFilters_path='../data/CNN_inputFilters',
            CNN_inputBias_path='../data/CNN_inputBias')

