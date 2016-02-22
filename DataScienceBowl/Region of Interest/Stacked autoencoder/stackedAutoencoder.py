
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import hiddenLayer as HL
import autoencoder as AC
import logisticRegression as LR
from scipy import misc
import matplotlib.pyplot as plt
import pickle



class StackedAutoEncoder(object):
    """Stacked denoising auto-encoder class (SdA)
    """

    def __init__(
        self,
        inputs,
        masks,
        numpy_rng,
        theano_rng=None,
        n_ins=4096,
        hidden_layers_sizes=[100, 100],
        n_outs=4096):

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
            self.X = inputs

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
                                        activation=theano.tensor.nnet.sigmoid)

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
    '''_
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
        n_ins=4096,
        hidden_layers_sizes=[100, 50, 30, 50, 100],
        n_outs=4096)

    for autoE in model_object.AutoEncoder_layers:
        # get the cost and the updates list
        cost, updates = autoE.get_cost_updates(**args)
        # compile the theano function
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={X: train_data[index * batch_size:(index + 1) * batch_size]})
        HL.iterate_epochs(n_epochs, numbatches, train_model, autoE)

    logReg = model_object.logLayer
    logcost, logupdates = logReg.get_cost_updates(**args)
    train_model = theano.function(
            inputs=[index],
            outputs=logcost,
            updates=logupdates,
            givens={X: train_data[index * batch_size:(index + 1) * batch_size],
                    Y: train_masks[index * batch_size:(index + 1) * batch_size]})

    HL.iterate_epochs(n_epochs, numbatches, train_model, logReg)


    return model_object



def finetune_sa(train_data, train_masks, numbatches, n_epochs, pretrainedSA, **args):

    '''
        Fine tunes stacked autoencoder
    '''
    finetunedSA = pretrainedSA

    traindim = train_data.shape
    batch_size = traindim[0]/numbatches


    index = T.lscalar()



    train_data = theano.shared(train_data)
    train_masks = theano.shared(train_masks)

    cost, updates = finetunedSA.get_cost_updates(**args)

    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                                  givens={finetunedSA.X: train_data[index * batch_size:(index + 1) * batch_size],
                                          finetunedSA.Y: train_masks[index * batch_size:(index + 1) * batch_size]})

    HL.iterate_epochs(n_epochs, numbatches, train_model, finetunedSA)

    return finetunedSA


def predict_sa(images, SA):


    mask_predictions = []

    for i in range(0, images.shape[0]):
        current_image = np.reshape(images[i, :, :], (1, (64*64)))
        predict_model = theano.function(
            inputs = [],
            outputs= SA.logLayer.y_pred,
            givens={SA.X: current_image})
        preds = predict_model()
        mask = np.reshape(preds, (64, 64))
        mask_predictions.append(mask)

    masks = np.array(mask_predictions)
    return masks


def crop_ROI(images, contours, roi, roi_dim, newsize):


    dim = images.shape
    image_roi = []
    contour_roi = []

    for i in range(0, dim[0]):

        # prep image files including up sampling roi to 256*256
        image = images[i, :, :]
        contour = contours[i, :, :]
        region = roi[i, :, :]
        region = misc.imresize(region, (dim[1], dim[2]))

        # get roi co-ords for cropping; using centre
        rows, cols = np.where(region == 1)
        cen_x, cen_y = (np.median(cols), np.median(rows))

        # execute  cropping on the image to produce ROI
        image = image[cen_x - (roi_dim[0]/2):cen_x + (roi_dim[0]/2),
                cen_y - (roi_dim[1]/2):cen_y + (roi_dim[1]/2)]

        image = misc.imresize(image, newsize)

        # execute cropping on the contour
        contour = contour[cen_x - (roi_dim[0]/2):cen_x + (roi_dim[0]/2),
                  cen_y - (roi_dim[1]/2):cen_y + (roi_dim[1]/2)]

        contour = misc.imresize(contour, newsize)

        # collect
        image_roi.append(image)
        contour_roi.append(contour)

    image_roi = np.array(image_roi)
    contour_roi = np.array(contour_roi)

    return image_roi, contour_roi

if __name__ == "__main__":

    # load required inputs and call training method (random data used until CNN is working)
    roi = np.load('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SBXtrainBinaryMask32')
    train = np.load('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SBXtrainImage256')
    mask = np.load('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SBXtrainMask256')

    print(roi.shape)
    plt.imshow(roi[1,:,:])
    plt.imshow(mask[1,:,:])
    plt.show()


    with open('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/CNN_output.pickle', 'rb') as f:
        roi_pred = pickle.load(f)
        roi_pred = np.asarray(roi_pred)
        thres = 0.5
        roi_pred[roi_pred >= thres ] = 1
        roi_pred[roi_pred < thres ] = 0

    train_roi, mask_roi =crop_ROI(images=train, contours=mask,
                                  roi=roi, roi_dim=(100,100), newsize=(64, 64))

    dim = mask_roi.shape
    print(mask_roi.shape)

    mask_roi = np.reshape(mask_roi, (dim[0], (dim[1]*dim[2])))
    mask_roi = np.array(mask_roi, dtype='float64')

    train_roi = np.reshape(train_roi, (dim[0], (dim[1]*dim[2])))
    train_roi= np.array(train_roi, dtype='float64')

    numbatches = 2
    batchdim = train.shape[0]/numbatches

    pretrainedSA = pretrain_sa(train_data=train_roi, train_masks=mask_roi, numbatches =numbatches,
                               n_epochs=10000, model_class=StackedAutoEncoder, datadim=batchdim,
                                            learning_rate=10, lam=10^4)

    finetunedSA = finetune_sa(train_data =train_roi, train_masks=mask_roi, numbatches =numbatches,
                               n_epochs=15000, pretrainedSA=pretrainedSA, datadim=batchdim,
                                            learning_rate=10, lam=10^4)


    images = np.load('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SBtrainImage64')

    mask_predictions = predict_sa(images, finetunedSA)
    mask_roi = np.reshape(mask_roi, (260, 64, 64))


    for i in range(0, 4):
        plt.subplot(1,5,1)
        plt.imshow(train[i,:,:])
        plt.subplot(1,5,2)
        plt.imshow(mask[i,:,:])
        plt.subplot(1,5,3)
        plt.imshow(misc.imresize(roi[i,:,:], (64,64)))
        plt.subplot(1,5,4)
        plt.imshow(mask_roi[i,:,:])
        plt.subplot(1,5,5)
        plt.imshow(mask_predictions[i,:,:])
        plt.show()

    with open('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SA_predHls', 'wb') as f:
        pickle.dump(mask_predictions, f)

    with open('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SA_modelHls', 'wb') as g:
        pickle.dump(finetunedSA, g)
