from __future__ import print_function
import os
import fnmatch
import re
import numpy as np
import dicom
#import cv2
from scipy import misc


# TODO add preprocessing on image data, like for kaggle training data LoadData routine
# TODO it has become apparant that the contour IDs do not map directly image IDs! Must fix code to deal with this


# Declare the top level directories that hold the image and contour files within the sunnybrook data

sb_root = "AML/data/SunnybrookData/"
image_dir = os.path.join(sb_root, "challenge_training")
contour_dir = os.path.join(sb_root,"Sunnybrook Cardiac MR Database ContoursPart3","TrainingDataContours")


# Two methods required, 1 for getting meta data about the relationship between the contour files and the image files,
# and the other to load images and masks into numpy array prior to saving to binary numpy files for easy re-use

def get_mapping(contour_dir):
    """
    gets the path, case and corresponding image mapping for each contour
    :param contour_dir: top level directory path where contours live

    RETURNS: arrays for contour path, contour series id and corresponding image id
    #NOTE: Consider to include CLASS for contour to clean up implementation in future.

    """
    c_path = []
    c_series = []
    c_imgid = []

    for root, _, files in os.walk(contour_dir):

         #Only interested in type 'i' images - representing endocardium. See sunnybrook 'README Contour Format ... . txt'.

        for f in fnmatch.filter(files, 'IM-0001-*-icontour-manual.txt'):
            path = os.path.join(root, f)
            c_path.append(path)

            # use regular expressions to match strings

            lookup = re.search(r"/([^/]*)/contours-manual/IRCCI-expert/IM-(\d{4})-(\d{4})-icontour-manual.txt", path)

            c_series.append(lookup.group(1)) # Contour series ID
            c_imgid.append("%s-%s" % (lookup.group(2), lookup.group(3)))  # Contour corresponding image ID based on parsed regular expression

    return c_path, c_series, c_imgid


def load_contours_dcm(c_path, c_series, c_imgid, image_dir):

    """
    loads data from DCM files into numpy arrays, and creates mask given contours
    :param c_path, c_series, c_imgid: array results from get_mapping
    :param image_dir: top level directory path where training images live

    RETURNS: numpy arrays for image data and corresponding masks built from contour co-ordinates
    """

    print('Source directory {0}...'.format(image_dir))

    imagedata = []
    contourdata= []

    for i in range(0, len(c_series)):

        file = "IM-%s.dcm" % (c_imgid[i])  #builds image path based on ID returned from contour
        path = os.path.join(image_dir, c_series[i], file)
        current_image = dicom.read_file(path)
        current_image = current_image.pixel_array.astype(np.int)
        contour = np.loadtxt(c_path[i], delimiter=" ").astype(np.int)
        mask = np.zeros_like(current_image, dtype="uint8")
        mask[contour] = 1
        ##cv2.fillPoly(mask, [contour], 1) #Need to install dependencies for cv2

        imagedata.append(current_image)
        contourdata.append(mask)


    imagedata = np.array(imagedata)
    contourdata = np.array(contourdata)

    print('Number of images {0}'.format(imagedata.shape[0]))

    return imagedata, contourdata


# Load data and store to numpy files for re-use

c_path, c_series, c_imgid = get_mapping(contour_dir)
imagedata, contourdata = load_contours_dcm(c_path, c_series, c_imgid, image_dir)

# numpy files will appear in data folder of directory

np.save('AML/data/SBtrainImage.npy', imagedata)
np.save('AML/data/SBtrainMask.npy', contourdata)


