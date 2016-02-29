# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 11:55:04 2016

@author: odyss
"""
import os
import re
import numpy as np
import pandas as pd
import dicom

class Patient(object):
    
    # static variable
    patient_count = 0
    
    def __init__(self, train_path, study):
        # intervening directories
        while True:
            subs = next(os.walk(train_path))[1]
            if len(subs) == 1:
                train_path = os.path.join(train_path, subs[0])
            else:
                break
        # now subs contains 'sax5', 'sax6', ...
                
        # list of height indices
        slices = []
        
        for sub in subs:
            m = re.match('sax_(\d+)', sub)
            if m is not None:
                slices.append(int(m.group(1)))
        
        # slices is a list containing [17,10,5,9,...,57] corresponding to sax
        
        slices_map = {}
        first = True
        times = []
        
        for cslice in slices:
            cslice_files = next(os.walk(os.path.join(train_path, 'sax_%d' % cslice)))[2]
            # cslice_files contains a list ['IM-4557-0021.dcm', 'IM-4557-0026.dcm',...]
            offset = None
            for cslice_file in cslice_files:
                m = re.match('IM-(\d{4,})-(\d{4})\.dcm', cslice_file)
                if m is not None:
                    if first:
                        times.append(int(m.group(2)))
                    if offset is None:
                        offset = int(m.group(1))
            first = False
            slices_map[cslice] = offset
            
        # some instance variables
        self.directory = train_path
        self.time = sorted(times)
        self.slices = sorted(slices)
        self.slices_map = slices_map
        Patient.patient_count += 1
        self.name = study
        
    # returns the name of a file of a specific slice on specific time
    def _filename(self, cslice, time):
        return os.path.join(self.directory, 'sax_%d' % cslice,
        'IM-%04d-%04d.dcm' % (self.slices_map[cslice], time))
        
    # read one single dicom file
    def _read_dicom_image(self, filename):
        d = dicom.read_file(filename)
        img = d.pixel_array.astype('int')
        return img
        
    # loads all patient's images. saves at instance variable self.images
    # images: [slice_height x time x 256 x 230]
    # images: [16 x 30 x 256 x 230]
    # also calculates slice thickness and area_multiplier (one for all images)
    def _read_all_dicom_images(self):
        f1 = self._filename(self.slices[0], self.time[0])
        d1 = dicom.read_file(f1)
        (x, y) = d1.PixelSpacing
        (x, y) = (float(x), float(y))
        f2 = self._filename(self.slices[1], self.time[0])
        d2 = dicom.read_file(f2)

        # try a couple of things to measure distance between slices
        try:
            dist = np.abs(d2.SliceLocation - d1.SliceLocation)
        except AttributeError:
            try:
                dist = d1.SliceThickness
            except AttributeError:
                dist = 8  # better than nothing...

        self.images = np.array([[self._read_dicom_image(self._filename(d, i))
                                 for i in self.time]
                                for d in self.slices])
        # sets dist equal to the two first slices distance (subtracting first
        # from second) or equal to first slice's thickness. I THINK the second
        # logic is better
        #
        # TODO
        #
        # maybe set dist = d1.SliceThickness
        #
                                
        self.dist = dist
        self.area_multiplier = x * y
    
    def load(self):
        self._read_all_dicom_images()
        
        
def calc_areas(images):
    # images are circular binary masks
    (slice_locations, times, _, _) = images.shape
    areas = [{} for t in range(times)]
    for t in range(times):
        for h in range(slice_locations):
            # TODO here, maybe we should feed the image to the model and then threshold the output
            areas[t][h] = np.count_nonzero(images[h][t])
    return areas
    
def calc_volume(areas, mult, dist):
    slices = np.array(sorted(areas.keys()))
    slices_new = [areas[h]*mult for h in slices]
    vol = 0
    for h in slices[:-1]:
        a, b = slices_new[h], slices_new[h+1]
        dvol = (dist/3.0) * (a + np.sqrt(a*b) + b)
        vol += dvol
    return vol / 1000.0
    
        
def calc_area_and_volume(patient):
    # compute the areas of all images. areas: [16 x 30] dictionary (I think)
    areas = calc_areas(patient.images)
    volumes = [calc_volume(area, patient.area_multiplier, patient.dist) for area in areas]
    # find edv and esv
    edv = max(volumes)
    esv = min(volumes)
    # calculate ejection fraction
    ef = (edv - esv) / edv
    
    # save as instance variables
    patient.edv = edv
    patient.esv = esv
    patient.ef= ef
        
# contains 'train', 'validate', etc
data_path = '/home/odyss/Documents/ucl/aml/kaggle/DataScienceBowl/data/'

labels = np.loadtxt(os.path.join(data_path, 'train.csv'), delimiter=',', skiprows=1)
label_dict = {}
for label in labels:
    label_dict[label[0]] = (label[2],label[1])

# here is the results of the CNN+SA+AC
#data_path = os.path.join(data_path, 'results')

# contains '1' (maybe patient '1')
train_path = os.path.join(data_path, 'train')
# contains 'sax5', 'sax6', ...
studies = next(os.walk(train_path))[1]

results_csv = open('results.csv', 'w')

for study in studies:
    patient = Patient(os.path.join(train_path, study), study)
    print 'Processing patient %s...' % patient.name
    try:
        patient.load()
        calc_area_and_volume(patient)
        (edv, esv) = label_dict[int(patient.name)]
        results_csv.write('%s,%f,%f,%f,%f\n' % (patient.name, edv, esv, patient.edv, patient.esv))
    except Exception as e:
        print '***ERROR***: Exception %s thrown by patient %s' % (str(e), patient.name)
    results_csv.close()
