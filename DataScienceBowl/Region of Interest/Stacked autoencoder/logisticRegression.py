import theano.tensor as T
import numpy as np
import timeit
from theano.tensor.shared_randomstreams import RandomStreams
import theano


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, masks, n_in, n_out):
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
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
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

        if masks is None:
            self.Y = T.dmatrix(name='masks')
        else:
            self.Y = masks

        if input is None:
            self.X = T.dmatrix(name='input')
        else:
            self.X = input

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(self.X, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]


    def get_cost_updates(self, datadim, learning_rate, lam):
        """
        :type scalar
        :param learning_rate: rate which weighs the gradient step

        :type scalar
        :param lam: regularization parameter for the cost function

        :type pair (cost, update)
        :return: compute cost and update for one training step of the autoencoder
        """

        # Compute the cost
        diff = self.y_pred - self.Y

        cost = T.true_div(T.nlinalg.trace(T.mul(diff, diff)), (2*datadim[0]))
                #+ T.nlinalg.norm(self.W)  # TODO add regularisation term

        # Compute updates
        gparams = T.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)]

        return (cost, updates)


def train_logreg(train_data, train_masks, numbatches,
                 n_epochs, model_class, **args):

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
    Y = T.matrix('Y')
    index = T.lscalar()

    train_data = theano.shared(train_data)
    train_masks = theano.shared(train_masks)


    model_object = model_class(
            input=X,
            masks=Y,
            n_in=100,
            n_out=4096)

    cost, updates = model_object.get_cost_updates(**args)

    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                                  givens={X: train_data[index * batch_size:(index + 1) * batch_size],
                                          Y: train_masks[index * batch_size:(index + 1) * batch_size]})


    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(n_epochs):
        for nindex in range(numbatches):
            c = train_model(nindex) #compute cost
            print 'Training epoch %d, batchId, cost' % epoch, nindex, c

    weights = model_object.W.get_value()
    bias = model_object.b.get_value

    return weights, bias

if __name__ == "__main__":

    # load sunny data and collapse to correct dim

    train = np.random.rand(1000, 10,10) #np.load('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SBtrainImage')
    trainMask = np.random.rand(1000, 64,64)#np.load('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SBtrainMask')
    dim = train.shape
    train = np.reshape(train, (dim[0], (dim[1]*dim[2])))
    train = np.array(train, dtype='float64')

    dim = trainMask.shape
    trainMask = np.reshape(trainMask, (dim[0], (dim[1]*dim[2])))
    trainMask = np.array(trainMask, dtype='float64')

    numbatches = 1
    batchdim = train[0]/numbatches

    final_weights, final_bias = train_logreg(train_data=train, train_masks=trainMask,
                                            numbatches=numbatches, n_epochs=10000,
                                            model_class=LogisticRegression, datadim=batchdim,
                                            learning_rate=10, lam=10^4)

