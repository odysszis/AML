import theano
from theano import tensor as T
import numpy as np
import HiddenLayer
import logisticRegression

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
        # after the construction of all class instances for sigmoid_layers and autoencoders is done we add a
        # logisticregression layer to represent the output layer
        # the instance logLayer of LogisticRegression has the following fields and methods
        #           - self.params
        #           - self.W
        #           - self.b
        #           - self.output
        #           - self.input <-- in constructor, this is initialized with the ouput of the last hidden (sigmoidal)
        #                            layer
        #           -
        self.logLayer = LogisticRegression(input = self.sigmoid_layers[-1].output,
                                            n_in = hidden_layers_sizes[-1],
                                            n_out = n_outs)
        self.params.extend(self.logLayer.params)
        self.finetune_cost = self.logLayer.cost_fn(self.y)

def test_sA(pretraining_epochs=15,
            training_epochs=1000,
            dataset,
            batch_size=1):
    dataset

