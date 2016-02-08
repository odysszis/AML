import timeit

import numpy
import pickle
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
import scipy.linalg.blas

from logisticReg import LogisticRegression, load_data
from LeNet import LeNetConvPoolLayer

def pre_training(learning_rate = 0.1, nkerns = 100, batch_size = 20):

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size                                           # 130

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

    # CNN + Pooling

    layer0_input = x.reshape((batch_size, 1, 64, 64))       # for now Fl and b0 are randomly initialized

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (64-11+1 , 64-11+1) = (54, 54)
    # maxpooling reduces this further to (54/6, 54/6) = (9, 9)
    # 4D output tensor is thus of shape (batch_size, 100, 9, 9)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        filter_shape=(nkerns, 1, 11, 11),
        image_shape=(batch_size, 1, 64, 64),                # 20 x 100 x 11 x 11
        poolsize=(6, 6)
    )

    layer0_output = layer0.output.flatten(2)                # 20 x 8,100

    # Logistic Regression Layer

    layer3 = LogisticRegression(input = layer0_output, n_in = 8100, n_out = 1024)

    layer3_output = layer3.output                           # 20 x 1024 tensor

    # cost for pre training

    # the cost we minimize during training is the NLL of the model

    cost = 0.5 / batch_size * T.sum( T.sum( layer3_output - y )**2 )    # 20x1024 - 20x1024
    cost += 0.9 / 2 * ( T.sum( T.sum( layer3.params[0] ** 2 ) ) )

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

    for minibatch_index in xrange(n_train_batches):

        if minibatch_index % 100 == 0:
            print('training @ iter = ', minibatch_index)
        cost_ij = train_model(minibatch_index)

    print('Optimization complete.')

    with open('preTrainLogReg.pickle', 'w') as f:
        pickle.dump([params], f)


if __name__ == '__main__':
    pre_training()