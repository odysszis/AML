import theano
from theano import tensor as T
import numpy as np
import HiddenLayer
import logisticRegression
import autoencoder


class sA(object):
    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            n_ins = 4096,
            hidden_layers_sizes = [100, 100],
            n_outs = 4096
    ):
        """
        :param numpy_rng: We need a random number generator for the simple autoencoders per hidden layer
                          to initialize Wvis, bvis
        :param theano_rng: see numpy_rng
        :param n_ins: region of interest has 4096 pixels (64 x 64)
        :param hidden_layers_sizes: the hidden layers have 100 units each
        :param n_outs: the LV shape has again 4096 pixels (64 x 64)
        """
        self.sigmoid_layers = [] # stores the sigmoid layers for each hidden layer
        self.dA_layers = []      # stores the autoencoders per hidden layer
        self.params = []         # stores all the weight matrices and bias vectors
        self.n_layers = len(hidden_layers_sizes) # the depth of the sA: number of hidden units
                                                 # we don't count the last logistic layer yet
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        # loop through the number of hidden layers and create a hidden-layer + autoencoder class instance
        # the hidden-layer is called sigmoid_layer because we use the sigmoid function as activation function
        # in these hidden layers
        for i in range(self.n_layers):
            if i == 0:
                input_size = n_ins
                layer_input = self.x
            else:
                input_size = hidden_layers_sizes[i-1]
                layer_input = self.sigmoid_layers[-1].output
            # create an instance of the HiddenLayer class which has following fields
            #       - self.input
            #       - self.W
            #       - self.b
            #       - self.params
            #       - self.output
            # It represents a sigmoidal layer with n_in input units and n_out output units.
            sigmoid_layer = HiddenLayer(rng = numpy_rng,
                                        input = layer_input,
                                        n_in = input_size,
                                        n_out = hidden_layers_sizes[i],
                                        activation = T.sigmoid)
            # Append the sigmoidal layer to our list of sigmoidal layers (there is i=0 and i=1, so we will have a list
            # of 2 sigmoidal layers)
            self.sigmoid_layers.append(sigmoid_layer)
            # Append the parameters self.W and self.b of the sigmoidal class
            self.params.extend(sigmoid_layer.params)
            # now create an instance of the class dA which has the following fields and methods
            #       - self.n_visible
            #       - self.n_hidden
            #       - self.Whid <-- in constructor, initialize Whid with W from HiddenLayer class instance sigmoid_layer
            #       - self.Wvis <-- in constructor, will be initialized as None
            #       - self.X
            #       - self.params
            #       - get_hidden
            #       - get_output
            #       - get_cost_updates
            dA_layer = dA(numpy_rng = numpy_rng,
                          theano_rng = theano_rng,
                          input = layer_input,
                          n_visible = input_size,
                          n_hidden = hidden_layers_sizes[i],
                          Whid = sigmoid_layer.W,
                          bhid = sigmoid_layer.b,
                          Wvis = None,
                          bvis = None)
            # add the dA_layer that represents an autoencoder at the current level i to the list of autoencoder
            self.dA_layers.append(dA_layer)
            # Append the parameters self.Whid, self.Wvis and self.bvis, self.bhid  of the autoencoder to the list of
            # parameters in the stacked autoencoder
            self.params.extend(dA_layer.params)
            # END OF FOR LOOP

        # after the construction of all class instances for sigmoid_layers and autoencoders is done we add a
        # logisticregression layer to represent the output layer
        # the instance logLayer of LogisticRegression has the following fields and methods
        #           - self.params
        #           - self.W
        #           - self.b
        #           - self.output
        #           - self.input <-- in constructor, this is initialized with the ouput of the last hidden (sigmoidal)
        #                            layer
        self.logLayer = logisticRegression(input = self.sigmoid_layers[-1].output,
                                            n_in = hidden_layers_sizes[-1],
                                            n_out = n_outs)
        self.params.extend(self.logLayer.params)
        self.finetune_cost = self.logLayer.get_cost_updates(self.y)

def train_sA(train_data, train_masks, numbatches,
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
            n_out=1024)

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

    # load required inputs and call training method (random data used until CNN is working)

    trainMask = np.random.rand(200, 32, 32)
    train = np.random.rand(200, 100)
    train = np.array(train, dtype='float64')

    dim = trainMask.shape
    trainMask = np.reshape(trainMask, (dim[0], (dim[1]*dim[2])))
    trainMask = np.array(trainMask, dtype='float64')
    numbatches = 1
    batchdim = train.shape[0]/numbatches

    final_weights, final_bias = train_logreg(train_data=train, train_masks=trainMask,
                                            numbatches=numbatches, n_epochs=1000,
                                            model_class=LogisticRegression, datadim=batchdim,
                                            learning_rate=10, lam=10^4)