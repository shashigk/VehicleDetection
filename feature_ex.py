import numpy as np
import cv2

from default_params import *
from utils import *

def extract_features_one_image (image,
                                color_space=DEFAULT_CSPACE,
                                spatial_size=DEFAULT_SPATIAL_SIZE,
                                hist_bins=DEFAULT_HIST_BIN,
                                hist_range=DEFAULT_HIST_RANGE,
                                orient=DEFAULT_ORIENT, 
                                pix_per_cell=DEFAULT_PIX_PER_CELL,
                                cell_per_block=DEFAULT_CELL_PER_BLOCK,
                                hog_channel=DEFAULT_HOG_CHANNEL,
                                spatial_feat=USE_SPATIAL_FEAT,
                                hist_feat=USE_HIST_FEAT,
                                hog_feat=USE_HOG_FEAT):
    file_features = []
    feature_image = convert2color (image, color_space)
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    if hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        file_features.append(hist_features)
    if hog_feat == True:
    # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        file_features.append(hog_features)
    return np.concatenate (file_features)

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs,
                     color_space=DEFAULT_CSPACE,
                     spatial_size=DEFAULT_SPATIAL_SIZE,
                     hist_bins=DEFAULT_HIST_BIN,
                     hist_range=DEFAULT_HIST_RANGE,
                     orient=DEFAULT_ORIENT, 
                     pix_per_cell=DEFAULT_PIX_PER_CELL,
                     cell_per_block=DEFAULT_CELL_PER_BLOCK,
                     hog_channel=DEFAULT_HOG_CHANNEL,
                     spatial_feat=USE_SPATIAL_FEAT,
                     hist_feat=USE_HIST_FEAT,
                     hog_feat=USE_HOG_FEAT):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = cv2.imread(file)
        file_features = extract_features_one_image (image, color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
        features.append(file_features)
    # Return list of feature vectors
    return features

def extract_features_from_frames (imgs, frames) :
    features = []
    if None != frames and len(frames) != len(imgs) :
        raise 'ERROR: Incompatible size of frames and images'
    for i in range(len(imgs)) :
        file_features = []
        image = cv2.imread (imgs[i])
        if frames != None :
            frame = frames[i]
            #print ("File = {} Frame [{}] = {}".format (imgs[i], i, frame))
            image = cv2.resize (image[frame[0][1]:frame[1][1], frame[0][0]:frame[1][0]], (64,64))
        file_features = extract_features_one_image (image)
        features.append(file_features)
    # Return list of feature vectors
    return features

    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            if startx >= endx or starty >= endy :
                continue #ignore invalid windows
            
            # Append window position to list
            window_list.append(((int(startx), int(starty)), (int(endx), int(endy))))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler,
                    color_space=DEFAULT_CSPACE,
                    spatial_size=DEFAULT_SPATIAL_SIZE,
                    hist_bins=DEFAULT_HIST_BIN,
                    hist_range=DEFAULT_HIST_RANGE,
                    orient=DEFAULT_ORIENT, 
                    pix_per_cell=DEFAULT_PIX_PER_CELL,
                    cell_per_block=DEFAULT_CELL_PER_BLOCK,
                    hog_channel=DEFAULT_HOG_CHANNEL,
                    spatial_feat=USE_SPATIAL_FEAT,
                    hist_feat=USE_HIST_FEAT,
                    hog_feat=USE_HOG_FEAT):
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = extract_features_one_image (test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            hist_range=hist_range,
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
