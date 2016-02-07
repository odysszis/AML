
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import hiddenLayer as HL
import autoencoder as AC
import logisticRegression as LR


class StackedAutoEncoder(object):
    """Stacked denoising auto-encoder class (SdA)
    """

    def __init__(
        self,
        inputs,
        masks,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10):

        self.sigmoid_layers = []
        self.AutoEncoder_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if masks is None:
            self.Y = T.dmatrix(name='masks')
        else:
            self.Y = masks
        if input is None:
            self.X = T.dmatrix(name='input')
        else:
            self.X = input

        for i in xrange(self.n_layers):

            # construct the sigmoidal layer
            # the size of the input
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer
            if i == 0:
                layer_input = self.X
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HL.HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            # Construct an autoencoder that shared weights with this layer and append to list
            AutoEncoder_layer = AC.AutoEncoder(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          Whid=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.AutoEncoder_layers.append(AutoEncoder_layer)

                    # We now need to add a logistic layer on top of the MLP
        self.logLayer = LR.LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            masks=self.Y,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)

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
        cost = T.mean((self.Y - self.logLayer.p_y_given_x) ** 2) # TODO: extend with regularisation terms

        # Compute updates
        gparams = T.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)]

        return (cost, updates)




def pretrain_sa(train_data, train_masks, numbatches, n_epochs, model_class, **args):
    '''
        Pretrains stacked autoencoder
    '''


    X = T.matrix('X')
    Y = T.matrix('Y')
    index = T.lscalar('index')

    traindim = train_data.shape
    batch_size = traindim[0]/numbatches

    train_data = theano.shared(train_data)
    train_masks = theano.shared(train_masks)

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    model_object = model_class(
        inputs=X,
        masks=Y,
        numpy_rng=rng,
        theano_rng=theano_rng,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10)

    for autoE in model_object.AutoEncoder_layers:
        # get the cost and the updates list
        cost, updates = autoE.get_cost_updates(**args)
        # compile the theano function
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={X: train_data[index * batch_size:(index + 1) * batch_size]})

        for epoch in xrange(n_epochs):
            for nindex in range(numbatches):
                c = train_model(nindex) #compute cost
                print 'Auto Encode Training epoch %d, batchId, cost' % epoch, nindex, c

    logReg = model_object.logLayer
    logcost, logupdates = logReg.get_cost_updates(**args)
    train_model = theano.function(
            inputs=[index],
            outputs=logcost,
            updates=logupdates,
            givens={X: train_data[index * batch_size:(index + 1) * batch_size],
                    Y: train_masks[index * batch_size:(index + 1) * batch_size]})

    for epoch in xrange(n_epochs):
        for nindex in range(numbatches):
            c = train_model(nindex) #compute cost
            print 'Log Reg Training epoch %d, batchId, cost' % epoch, nindex, c


    return model_object



def finetune_sa(train_data, train_masks, numbatches, n_epochs, pretrainedSA, **args):

    '''
        Fine tunes stacked autoencoder
    '''
    finetunedSA = pretrainedSA

    traindim = train_data.shape
    batch_size = traindim[0]/numbatches

    X = T.matrix('X')
    Y = T.matrix('Y')
    index = T.lscalar()

    train_data = theano.shared(train_data)
    train_masks = theano.shared(train_masks)

    cost, updates = finetunedSA.get_cost_updates(**args)

    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                                  givens={X: train_data[index * batch_size:(index + 1) * batch_size],
                                          Y: train_masks[index * batch_size:(index + 1) * batch_size]})

    for epoch in xrange(n_epochs):
        for nindex in range(numbatches):
            c = train_model(nindex) #compute cost
            print 'Training epoch %d, batchId, cost' % epoch, nindex, c

    return finetunedSA


if __name__ == "__main__":

    # load required inputs and call training method (random data used until CNN is working)

    trainMask = np.random.rand(4000, 32, 32)
    train = np.random.rand(4000, 100)
    train = np.array(train, dtype='float64')

    dim = trainMask.shape
    trainMask = np.reshape(trainMask, (dim[0], (dim[1]*dim[2])))
    trainMask = np.array(trainMask, dtype='float64')

    numbatches = 1
    batchdim = train.shape[0]/numbatches

    pretrainedSA = pretrain_sa(train_data=train, train_masks=trainMask, numbatche =numbatches,
                               n_epochs=10, model_class=StackedAutoEncoder, datadim=batchdim,
                                            learning_rate=10, lam=10^4)

    finetunedSA = finetune_sa(train_data =train, train_masks=trainMask, numbatche =numbatches,
                               n_epochs=10, pretrainedSA=pretrainedSA, datadim=batchdim,
                                            learning_rate=10, lam=10^4)

