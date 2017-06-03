---

**Vehicle Detection Project**

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[imageCar5x5]:   ./output_images/car_5x5.png
[imageCarYCrCb]: ./output_images/car_YCrCb_5x5.png
[imageCarHog0]:  ./output_images/car_hog0.png
[imageCarHog1]:  ./output_images/car_hog1.png
[imageCarHog2]:  ./output_images/car_hog2.png
[imageHogAll]:   ./output_images/hog_features_img.jpg
[imageColorHis]: ./output_images/colorspace_histogram.jpg
[imageSpatBin]:  ./output_images/spatialbin_histogram.jpg
[imageNoCar]:    ../data/non-vehicles/GTI/image1000.png
[imageCar]:      ../data/vehicles/GTI_Far/image0001.png 
[imageCarDetMW]: ./output_images/car_detection.png
[imageCarDetHM]: ./output_images/car_heatmap.png
[imageCarSinHM]: ./output_images/car_heatwin_sw_falsepos.png
[imageCarMulWi]: ./output_images/car_multiwins.png
[imageCarSinWi]: ./output_images/car_singlewin.png
[imageCarSinFP]: ./output_images/car_singlewin_falsepos.png
[imageCarMwFPNx]: ./output_images/car_multiwin_nox_FP_t5.png
[imageNCHM]: 	 ./output_images/nocar_heatmap.png
[imageNCMulWi]:  ./output_images/nocar_multiwins.png
[imageNCNoDet]:  ./output_images/nocar_nodetection.png
[imageNCSinWi]:  ./output_images/nocar_singlewin.png
[imageYCrCbFea]: ./output_images/ycrcb_feature_img.jpg
[imageSlWinBef]: ./output_images/sliding_win_bef_tune.png
[imageSlWinBRe]: ./output_images/sliding_win_bef_tune_res.png
[imageSlWinAft]: ./output_images/sliding_win_aft_tune.png
[imageSlWinAre]: ./output_images/sliding_win_aft_tune_res.png
[imageT5Hmap]:   ./output_images/test5_hotmap.png
[imageT5Hwin]:   ./output_images/test5_hotwins.png
[video1]: ./project_video_out.mp4

---
### README

#### Project Files

The main files are :
* Project\_Report.md -- The README for the project
* Classifier.ipynb -- Notebook containing code for feature extraction and classifier training.
* Detect.ipynb -- Notebook containing code for car detection in images and videos.
* FeatureVis.ipynb  -- Utility notebook to visualize hog, spatial and color histogram features for the report.
* project\_video\_out.mp4 -- Output Project video

The rest of the code is in python files below:

* feature\_ex.py -- Functions to extract HOG and color and spatial features.
* default\_params.py -- Parameters for the project e.g., color space, hog feature parameters etc.
* heat.py  -- Functions for adding heat maps.
* utils.py -- Utilities e.g., color conversion.
* moredata.py  -- Functions to udacity augmented data (unused).

The model and scaler are saved as follows:
* vehicle\_model.pkl -- The SVM model saved as a pickle file
* vehicle\_scaler.pkl -- The Standard scaler saved as a pickle file


### Acknowlegement

I thank the previous reviewer and forum mentors for providing valuable feedback, particularly, involving
the need to convert to BGR space for video pipeline. While, I appreciate the previous reviewer's feedback,
the color conversion was the main reason for unstable windows in my previous implementation.

A lot of the code has been used from the class lectures. The idea of smoothing out the heatmaps 
was discussed in a forum and I based my implementation on that.

### Feature Extraction And Classifier Training

#### 1. Data Set Characteristics and Feature Vector Composition

The images in the `vehicle` and `non-vehicle` dataset were used for training a classifier.

An example of a vehicle image is below.

![imageCar][imageCar]

An example of a non-vehicle image is below.

![imageNoCar][imageNoCar]

Throughout the code I used OpenCV API cv2.imread to read the test images.
This API reads the images in BGR colorspace. I used YCrCb color spaces, 
I saw that YCrCb seemed to work better on some test images and
decided to use it. Further, the vehicles can appear in any color, so color spaces such
as HSV may not work well. An example image in YCrCb is shown below:

![imageYCrCbFea][imageYCrCbFea]


For feature selection, I used the following parameters:
* Spatial binning of color with size `(32, 32)`.
  An example histogram of spatial features for an image from training set is shown below:
  
  ![imageSpatBin][imageSpatBin]

* Color histogram features, with `32 bins` and `range (0, 256)`.
 
  An example histogram of these features is shown below:
 
  ![imageColorHis][imageColorHis]

* HOG features, with parameters `orientations = 9`, `pixels_per_cell = 8` and `cells_per_block = 2`.
  The hog features were computed using `skimage.hog()`.

  The Hog features were used on all color channels. The following is a representation for each channel.

  ![imageHogAll][imageHogAll]

The code for feature extraction is in the file `feature_ex.py`.

#### 2. Linear SVM Classifier

There were about 8792 vehicle images, and 8968 non-vehicle images.
After feature extraction, each image resulted in a feature vector with 8460 features.

The featuers were normalized using `StandardScaler`.

The features were split randomly in the ratio 80% / 20 % into training and validation sets.
This data set was then used to train a linear SVM classifier using `LinearSVC` from `sklearn.svm`.
The models and the scaler were the saved into 'vehicle\_model.pkl', and 'vehicle\_scaler.pkl' respectively.

With the features selected as above I could train the classifier with an accuracy of `99.1%` on
the validation set. This was high enough accuracy and I did not explore other parameter settings.

The code for classifier training is part of Classifier.ipynb.

### Sliding Window Search

#### 1. Sliding Window Implementation and Tuning.

In order to detect vehicles in the images, I used a sliding window approach.
In this approach windows of various sizes were used to compute subsets of the image
and resized to (64,64) size. Feature extraction as explained above was then applied
the trained classifier was used to predict whether a vehicle was present in a car.

The major part of the project involved properly implementing the sliding window search
for cars in images. The goal was to minimize the false positives and at the same time
make sure that all the cars were detected in all images and frames of the videos.

For example, in my initial implementation, I used four groups of sliding windows
namely, with y-coordinate ranges and overlaps shown below.

|Window Scale| (y\_start, y\_stop) | Overlap |
:------------|---------------------|---------|
|32x32	     | (400, 528)          | None    |
|32x32       | (500, 628)          | 25%     |
|96x96       | (400, 656)          | 50%     |
|128x128     | (400, 656)          | 50%     |

An image with these windows draw is illustrated below:

![imageSlWinBef][imageSlWinBef]

Using the raw windows with positive classification, this resulted in many false positives, as shown below.

![imageCarMwFPNx][imageCarMwFPNx]

I then created a heatmap and applied a threshold to ensure at least two groups were
overalapping and this eliminated the false positives. However, there were images,
where the white car was not detected; see figure below.

![imageSlWinBRe][imageSlWinBRe]

After analyzing I found that it was not a problem with the classifier per se, 
but rather a problem of incorrect offset
of sliding windows. After a few experiments, I finally arrived at the following offset
and sizes for the sliding windows.

|Window Scale| (y\_start, y\_stop) | Overlap |
:------------|---------------------|---------|
|96x96	     | (400, 656)          | 50%     |
|96x64       | (404, 468)          | 50%     |
|128x128     | (400, 656)          | 50%     |
|128x64      | (404, 468)          | 50%     |


The detailed code for creating these sliding windows is in the 
function `create_sliding_windows` in the notebook `Detect.ipynb`.

An image with these windows is illustrated below:

![imageSlWinAft][imageSlWinAft]

Using these windows there were no missing cars in the images, as confirmed by the example below.

![imageSlWinAre][imageSlWinAre]


Also, the corresponding hot windows are shown below.

![imageT5Hwin][imageT5Hwin]

The *thresholded* heatmap is shown below:

![imageT5Hmap][imageT5Hmap]


The illustrations for test images are all in the notebook Detect.ipynb.
This image pipeline is implemented as a function, namely, `process_image` in Detect.ipynb.

#### 2. Video Pipeline

The video pipeline implementation involved invoking the image pipeline functionality
with each frame of the video. However, there were two modifications:
1. Before feeding the frame to image pipeline, it was converted to BGR space.
2. A moving average heatmap was maintained to have smooth vehicle detection.

The second part, namely, moving average heatmap was implemented as class `VehicleTracker`
in Detect.ipynb. As suggested by discussions in the forum, I used a deque of size 10
to maintain the heatmap of the last 10 frames and then thresholded the sum. The
details are in the member function `process_frame` of the `VechicleTracker` class.

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

I was successfully able to eliminate the false positives and detect cars in all frames
of the videos using the methods described above. However, a number of improvments are 
possible, namely:
1. Make classifier robust to different search with different window sizes and offsets. 
   One potential way to achieve this is to augment the training data with various cropped portions of the images, with
   cropping corresponding to window choices and overlaps.
2. Use brightness augmentation of the data. I noticed the training images have different lighting to the video.
3. Another aspect apart from accuracy to improve would be runtime. One obvious direction is to use HOG subsampling method.


