import numpy as np
import cv2
from skimage.feature import hog

from default_params import *

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack ((color1, color2, color3))

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features


def select_subset (size, lst) :
    return list[0:min(len(lst), size)]

def select_random_subset (size, lst) :
    if len(lst) <= size :
        return lst
    rand_inds = np.random.choice (len(lst), size, replace=False)
    return lst[rand_inds]

# Select a random subset of features and labels.
def select_random_subset_data (sample_size, X, y) :
    if (len(X) != len(y)) :
        raise 'ERROR: select_subset, incompatible features and label sizes'
    if (len(X) <= sample_size) :
        return X, y
    rand_inds = np.random.choice(len(X), sample_size, replace=False)
    return X [rand_inds], y [rand_inds]

# How to use shuffle, check example below.
#from sklearn.utils import shuffle
#X, y = shuffle (X, y)


def convert_color(img, conv) :
    conv = conv.upper()
    if conv == 'RGB2YCRCB':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'BGR2YCRCB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'BGR2LUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    if conv == 'BGR2HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if conv == 'BGR2HLS':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    print ("conv = {}".format (conv))
    raise 'ERROR: convert_color, unsupported color conversion requested {}'.format (conv)

def convert2color (image, dstcolor) :
    if dstcolor == NATIVE_COLOR_SPACE :
        return np.copy (image)
    if NATIVE_COLOR_SPACE != 'BGR' :
        raise 'ERROR: Only BGR native color space is supported'
    return convert_color (image, '{}{}{}'.format (NATIVE_COLOR_SPACE, '2', dstcolor))


