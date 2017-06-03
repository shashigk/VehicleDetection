import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split


from default_params import *
from utils import *
from feature_ex import *
from moredata import *

# Divide up into cars and notcars
# Read in car and non-car images
cimages = glob.glob('../data/vehicles/GTI_Far/*.png')
cars = []
totalCount = 40
for image in cimages :
    cars.append (image)
    if (len(cars) >= totalCount) :
        break

ncimages = glob.glob('../data/non-vehicles/GTI/*.png')
notcars = []
for image in ncimages :
    notcars.append (image)
    if (len(notcars) >= totalCount) :
        break


print ("carimgs = {}, notcarimgs = {}".format (np.shape (cars), np.shape(notcars)))
morecars, morecarbbs, morenotcars, morenotcarbbs = LoadExtraData ()



def train_with_features () :
    # Reduce the sample size because HOG features are slow to compute
    # The quiz evaluator times out after 13s of CPU time
    #### sample_size = 500
    #### cars = cars[0:sample_size]
    #### notcars = notcars[0:sample_size]
    
    ### TODO: Tweak these parameters and see how the results change.
    
    t=time.time()
    car_features = extract_features(cars)
    notcar_features = extract_features(notcars)
    morecar_feats = extract_features_from_frames (morecars, morecarbbs)
    morenotcar_feats = extract_features_from_frames (morenotcars, morenotcarbbs)
    print ("Car Features = {} ".format (np.shape (car_features)))
    print ("Not Car Features = {}".format (np.shape (notcar_features)))
    car_features = car_features + morecar_feats
    notcar_features = notcar_features + morenotcar_feats
    print ("More Car Features = {} ".format (np.shape (car_features)))
    print ("More Not Car Features = {}".format (np.shape (notcar_features)))


    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    
    print('Using:',DEFAULT_ORIENT,'orientations',DEFAULT_PIX_PER_CELL,
        'pixels per cell and', DEFAULT_CELL_PER_BLOCK,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

train_with_features ()

