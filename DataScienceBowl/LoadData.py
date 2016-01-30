from __future__ import print_function
import os
import numpy as np
import dicom
#from scipy.misc import imresize #for use with a crop pre process


# TODO use pickle for object serialisation
# TODO reshape data to collapse into array of pixels
# TODO add different preprocessing methods
# TODO add method for storing resulting numpy arrays as theano shared variables
# TODO add method for mapping input images to ground truth masks
# TODO include meta data within download

# local test: data = load_dcm_data('/Users/Peadar/Documents/KagglePythonProjects/AML/data/validate',0)


def load_dcm_data(directory, preprocess):

    """
    :param directory: top level folder where image .dcm files are stored
    """
    print('Loading from {0}...'.format(directory))

    imagedata = [] #placeholder for imagedata array

    for root, _, files in os.walk(directory):
        total = 1

        for f in files:
            image_path = os.path.join(root, f)
            if not image_path.endswith('.dcm') or root.find("sax") == -1: #sax files are of interest initially as per kaggle site
                continue
            current_image = dicom.read_file(image_path)
            current_image = current_image.pixel_array.astype(float)/np.max(current_image.pixel_array) #between 0 and 1
            #current_image = preprocess(current_image) #In case there is any image level preprocessing required like cropping
            imagedata.append(current_image)
            print('Images loaded {0}'.format(total))
            total += 1

    imagedata = np.array(imagedata, dtype=np.uint8)
    print('Number of images {0}'.format(imagedata.shape[0]))
    return imagedata


def cropImage():

    """
    TBC
    """
    #EG for things like cropping etc
    return 0



