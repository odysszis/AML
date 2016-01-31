from __future__ import print_function
import os
import numpy as np
import dicom
from scipy.misc import imresize


# TODO use pickle for object serialisation
# TODO reshape data to collapse into array of pixels
# TODO add method for storing resulting numpy arrays as theano shared variables
# TODO add method for mapping input images to ground truth masks
# TODO include meta data within download

# LOCAL TEST: data = load_dcm_data('/Users/Peadar/Documents/KagglePythonProjects/AML/data/validate', crop_resize, newsize = (64,48))


def load_dcm_data(directory, preprocess, **args):

    """
    :param directory: top level folder where image .dcm files are stored
    :param preprocess: function handle which preprocessing method to apply to images
    :param **args: any named arguments and their values required by preprocess method
    """
    print('Loading from {0}...'.format(directory))

    imagedata = [] #placeholder for imagedata array

    for root, _, files in os.walk(directory):
        total = 1

        for f in files:
            image_path = os.path.join(root, f)

            if not image_path.endswith('.dcm') or root.find("sax") == -1: #sax files are of interest initially as per data notes on kaggle
                continue

            current_image = dicom.read_file(image_path)
            current_image = current_image.pixel_array.astype(float)/np.max(current_image.pixel_array) #between 0 and 1
            current_image = preprocess(current_image, **args) #applies selected prepocess routine

            imagedata.append(current_image)

            print('Images loaded {0}'.format(total))
            total += 1

    imagedata = np.array(imagedata)

    print('Number of images {0}'.format(imagedata.shape[0]))

    return imagedata



def crop_resize(image, crop=False, newsize=()):

    """
    :param image: numpy array to be processed
    :param crop: true/false whether to crop image from centre
    :param newsize: optional tuples for new image size

     returns cropped and resized image
    """
    if crop:
        edge = min(image.shape[:2]) #cropping from centre by using half the short edge length
        y = int((image.shape[0] - edge) / 2)
        x = int((image.shape[1] - edge) / 2)
        image = image[y: y + edge, x: x + edge]

    image = imresize(image, newsize)  #using scipy resize function tuple for new size

    return image



