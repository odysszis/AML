from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import active_contour as AC
import sys
sys.path.insert(0, '/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/Region of Interest/Stacked autoencoder/')
import stackedAutoencoder
from stackedAutoencoder import crop_ROI



"""
    This module runs validation for best parameter selection against the active contour model using predictions from the stacked autoencoder.
    It uses the validation function from within the active_contour module
"""

# ADD YOUR DATA'S LOCAL DATA PATH
LOCALDATAPATH = '/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/'

# load required data

# binary region of interest square (output from CNN)
roi = np.load(LOCALDATAPATH + 'SBXtrainBinaryMask32')

# original training image
train = np.load(LOCALDATAPATH + 'SBXtrainImage256')

# original training binary contour
mask = np.load(LOCALDATAPATH + 'SBXtrainMask256')

# predictions from SA based on the above data
preds = np.load(LOCALDATAPATH + 'SA_predictions')
preds = preds[0:2, :, :]
# crop original image and contour data to match the region of the predictions
train_roi =crop_ROI(images=train, roi=roi, roi_dim=(100,100), newsize=(64, 64))
train_roi = train_roi[0:2, :, :]
mask_roi =crop_ROI(images= mask, roi=roi, roi_dim=(100,100), newsize=(64, 64))
mask_roi = mask_roi[0:2, :, :] #add subset

"""
    Trial parameter ranges: alpha1{1, 1.5, 2}, alpha 2{1.5,2,2.5},  alpha 3 = {0, ..., 0.01} steps 0.001

"""

#params are [alpha1 set, alpha 2 set, alpha3 set]

params = [[1, 1.5,2], [1.5, 2, 2.5], [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]]

# run the validation function to find best set of parameters from the combinations available in the above list

best_params = AC.ac_val(preds, train_roi, mask_roi, params)

best_params.dump(LOCALDATAPATH + 'AC_params')