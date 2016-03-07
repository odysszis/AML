from __future__ import print_function
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
import os
from stackedAutoencoder import SA
from stackedAutoencoder import crop_ROI



def inspect_inputs(i, node, fn):
    print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs],
          end='')

def inspect_outputs(i, node, fn):
    print(" output(s) value(s):", [output[0] for output in fn.outputs])


def predict_sa(images, trained_SA_path = '/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SA_X10modelCPU'):



    with open(trained_SA_path) as f:
         SA_inst = pickle.load(f)

    dim = images.shape
    images = np.reshape(images, (dim[0], (dim[1]*dim[2])))

    mask_predictions = []

    predict_model = theano.function(
            inputs = [SA_inst.X],
            outputs= SA_inst.logLayer.p_y_given_x)
            #mode=theano.compile.MonitorMode(
                        #pre_func=inspect_inputs,
                        #post_func=inspect_outputs))

    for i in range(0, dim[0]):
        current_image = np.reshape(images[i,:], ((dim[1]*dim[2]), 1))
        pred = predict_model(np.transpose(current_image))
        mask_predictions.append(pred)

    mask_predictions = np.reshape(mask_predictions, (dim[0], dim[1], dim[2]))
    images = np.reshape(images, (dim[0], dim[1], dim[2]))
    masks = np.array(mask_predictions)


    return masks

roi = np.load('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SBXtrainBinaryMask32')
train = np.load('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SBXtrainImage256')
mask = np.load('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SBXtrainMask256')
train_roi =crop_ROI(images=train, roi=roi, roi_dim=(100,100), newsize=(64, 64))
mask_roi =crop_ROI(images= mask, roi=roi, roi_dim=(100,100), newsize=(64, 64))
mask_predictions = predict_sa(train_roi, trained_SA_path = '/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SA_RLUmodel')


for i in range(0,100):
    f, axarr = plt.subplots(3, 2)
    axarr[0, 1].imshow(train[i,:,:])
    axarr[0, 1].set_title('Original Image')
    axarr[0, 0].imshow(mask[i,:,:])
    axarr[0, 0].set_title('Original Contour')
    axarr[1, 0].imshow(roi[i,:,:])
    axarr[1, 0].set_title('Original binary ROI')
    axarr[1, 1].imshow(train_roi[i,:,:])
    axarr[1, 1].set_title('Original Image ROI')
    axarr[2, 0].imshow(mask_roi[i,:,:])
    axarr[2, 0].set_title('Original Contour ROI')
    axarr[2, 1].imshow(mask_predictions[i,:,:])
    axarr[2, 1].set_title('Predicted Contour')
    plt.show()