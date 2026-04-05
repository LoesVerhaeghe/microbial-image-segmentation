import os
import sys
sys.path.append("..")

# basic imports for visualization, image loading and classification
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skimage import io

import numpy as np

# libraries used in functions
from scipy.ndimage import distance_transform_edt


def create_buffer_background_image(mask, buffer_radius=15, no_feature_radius=100):
    """
    Function to create a segmented image distinguishing: feature (1), no-feature (2)
    and unknown (0) where unknown is a buffer zone around the features
    
    PARAMETERS
    ----------
    mask : np.array
        binary image where feature pixels are one
    
    buffer_radius : int
        size of buffer zone around feature pixels

    no_feature_radius : int
        all pixels between buffer_radius and no_feature_radius are used as training data
        instances for 'no feature' class (gets label 2). If None, it is set to nr of rows
        in the image meaning that it is ignored. Default is 100
    
    RETURNS
    -------
    buffer_mask : np.array 
    """
    # If no_root_radius is None, set it to the number of rows in the image
    # -> ensures the radius is large enough to include the whole image
    if no_feature_radius is None:
        no_feature_radius = mask.shape[0]

    # Compute the Euclidean distance transform (EDT) of the inverted mask
    # ~root_mask -> background becomes True (0->1), roots become False (1->0)
    # dt[x,y] = distance from pixel (x,y) to the nearest root pixel
    dt = distance_transform_edt(mask==0)

    # Define "no-feature" pixels:
    # - farther away than buffer_radius from features (outside the uncertain zone)
    # - but not farther than no_root_radius (so we don't label infinite background)
    no_feature_mask = (dt > buffer_radius) & (dt < no_feature_radius)

    # Start with a copy of root_mask as uint8
    # - features are 1, everything else 0
    buffer_mask = np.array(mask, dtype=np.uint8)

    # Set the "no-feature" pixels (from mask above) to label 2
    buffer_mask[no_feature_mask] = 2

    # Return the final segmented image:
    # - 1 = feature
    # - 2 = no-feature
    # - 0 = buffer (unknown zone near features)
    return buffer_mask



#######################################################################################################
# test the buffer making function


# import cv2

# # Read as grayscale (values 0–255)
# img = cv2.imread("images/masks/task-1-annotation-1-by-1-tag-filament-0.png", cv2.IMREAD_GRAYSCALE)
# mask = (img > 0).astype(np.uint8)# Binarize (turn into 0/1 mask)

# orig = cv2.imread("images/original/04152749.JPG")        # BGR format
# orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)  # convert to RGB for matplotlib

# # Make an overlay for plotting
# overlay = orig.copy()
# overlay[mask == 1] = [255, 0, 0]  # paint mask pixels red
# alpha = 0.5  # transparency: 0 = only original, 1 = only mask
# blended = cv2.addWeighted(orig, 1 - alpha, overlay, alpha, 0)

# fig = plt.figure(figsize=(12, 4), dpi=250)
# plt.imshow(blended)
# plt.axis("off")
# plt.show()

# ### create buffer around filaments
# buffer_mask=create_buffer_background_image(mask, buffer_radius=1, no_feature_radius=1000)

# from matplotlib.colors import ListedColormap
# cmap = ListedColormap(["purple", "blue", "yellow"])  
# fig = plt.figure(figsize=(12, 4), dpi=250)
# plt.imshow(buffer_mask,  cmap=cmap, vmin=0, vmax=2)
# plt.axis("off")
# plt.show()


# overlay2 = orig.copy()
# overlay2[buffer_mask == 1] = [255, 0, 0]  # paint mask pixels red
# overlay2[buffer_mask == 0] = [0, 255, 0]
# alpha = 0.5  # transparency: 0 = only original, 1 = only mask
# blended2 = cv2.addWeighted(orig, 1 - alpha, overlay2, alpha, 0)

# fig = plt.figure(figsize=(12, 4), dpi=250)
# plt.imshow(blended2)
# plt.axis("off")
# plt.show()



##########################
#make features and labels

def get_save_fname(fname, save_dir, suffix = ".dummy"):
    """
    Function to split the absolute path string into directory and file name and append a suffix.
    If save_dir is None, the directory will be joined with the file name where the extension (after the last .)
    is replaced with `suffix`. Else, the directory of the file name will be replace with `save_dir`.
    
    PARAMETERS
    ---------
    fname : str
        Absolute path string of the file
    save_dir : str
        Directory where the file will be saved
    suffix : str
        Suffix to be added to the file name
        
    RETURNS
    -------
    str
        Absolute path string of the saved file
    """
    directory, filename = os.path.split(fname)
    if save_dir is None:
        return os.path.join(directory, f"{filename.split('.')[0]}_{suffix}")
    return os.path.join(save_dir, f"{filename.split('.')[0]}_{suffix}")


from skimage import draw, morphology, feature
from functools import partial
from skimage.io import imread
from PIL import Image
import traceback

def im2features(im, sigma_min=1, sigma_max=4):
    """
    Simple wrapper around feature.multiscale_basic_features to compute 
    features of a given image
    
    PARAMETERS
    ----------
    im : np.array
        RGB image
    
    sigma_min : int
        minimal value for smooting kernel bandwidth
        
    sigma_max : int
        maximal value for smooting kernel bandwidth
    
    RETURNS
    -------
    np.array with size (nrow x ncol x k) where k is the number of features
    """
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=True, texture=True,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            channel_axis=-1) #intensity based features, edge based features, texture based features
    return features_func(im)


def imgs_to_XY_data(orig_img_list : list[str] = None,
                    masks_file_list : list[str] = None,
                    buffer_radius : int = 1,
                    no_feature_radius : int = 1000,
                    sigma_max : int = 4,
                    save_dir : str = './',
                    palette=None,
                    save_masks_as_im : bool = False) -> None:
    """
    Function to transform a set of images and masks to Features (X) and Labels (Y)
    that can be used for training a segmentation model. Computed Features (X) and Labels (Y) are stored as
    .npy files. 
    
    RETURNS
    -------
    None
    """

    for orig_img_file, mask_file in zip(orig_img_list, masks_file_list):            
            # file name for saving (if None save in same directory as reading)
            features_fname = get_save_fname(orig_img_file, save_dir, "FEATURES.npy")
            labels_fname = get_save_fname(orig_img_file, save_dir, "LABELS.npy")
            img_fname = get_save_fname(orig_img_file, save_dir, "MASK.png")

            if not os.path.isfile(features_fname):
                try:
                    # create mask with buffer
                    mask = imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    mask = (mask > 0).astype(np.uint8)# Binarize (turn into 0/1 mask)
                    buffer_mask = create_buffer_background_image(mask, buffer_radius = buffer_radius, no_feature_radius = no_feature_radius)

                    # create training data labels
                        # - 1 = feature
                        # - 2 = no-feature
                        # - 0 = buffer (unknown zone near features)
                    training_labels = buffer_mask

                    # compute features
                    orig_img=imread(orig_img_file)
                    features = im2features(orig_img, sigma_max = sigma_max)

                    # flatten labels and features
                    features_flat = features[training_labels > 0,:] #because if trainingslabel is zero, it's unsure which class it iss
                    label_flat = training_labels[training_labels > 0]

                    # save features
                    np.save(features_fname, features_flat)
                    np.save(labels_fname, label_flat)

                    if save_masks_as_im:
                        # save segmentation mask as image
                        pi = Image.fromarray(training_labels,'P')
                        # Put the palette in
                        pi.putpalette(palette)
                        # Display and save
                        pi.save(img_fname)

                except Exception as e:
                    print("Problem processing " + orig_img)
                    print("Traceback of -- ", orig_img )
                    traceback.print_exc()
                    print("End traceback -- ", orig_img )
            else:
                print("skipping " + orig_img_file, "as it FEATURES exist already")



# a palette for saving segmentation masks as rgb images
# Make a palette
palette = [255,0,0,    # 0=red
           0,255,0,    # 1=green
           0,0,255,    # 2=blue
           255,255,0,  # 3=yellow
           0,255,255]  # 4=cyan
# Pad with zeroes to 768 values, i.e. 256 RGB colours
palette = palette + [0]*(768-len(palette))


image_files = [f for f in os.listdir('images/original')]
images_list = [os.path.join('images/original', f) for f in image_files]
mask_files = [f for f in os.listdir('images/masks')]
masks_list = [os.path.join('images/masks', f) for f in mask_files]

imgs_to_XY_data(orig_img_list =images_list,
                    masks_file_list=masks_list,
                    buffer_radius = 1,
                    no_feature_radius = 1000,
                    sigma_max = 4,
                    save_dir = 'features/',
                    palette=palette,
                    save_masks_as_im = True) 

###################################################


# combine the generated files to create X and Y

def compile_training_dataset_from_precomputed_features(features_file_list: list[str] = None,
                                                       labels_file_list: list[str] = None,
                                                       sample_fraction: tuple[float, float] = (0.3, 0.1),
                                                       seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to compile a training set from a set of FEATURES and LABELS npy-files by
    loading each file, taking a sample of relative size sample_fraction[0] for features and
    sample_fraction[1] for background.

    PARAMETERS
    ----------
    features_file_list : list[str], optional
        List of file names of features npy-files. If not provided, default files will be used.
    labels_file_list : list[str], optional
        List of file names of labels npy-files. If not provided, default files will be used.
    sample_fraction : tuple[float, float], optional
        Tuple of two floats representing the relative size of the sample for features and background.
        Default is (0.3, 0.1).
    seed : int, optional
        Seed value for random number generation. Default is 1.

    RETURNS
    -------
    tuple[np.ndarray, np.ndarray]
        The first output is the X set (features), and the second output is the Y set (labels).
    """
    np.random.seed(seed)

    features_list = []
    labels_list = []

    for feature_file, label_file in zip(features_file_list, labels_file_list):
        X = np.load(feature_file) 
        Y = np.load(label_file)

        n_1 = np.sum(Y == 1) #number of pixels belonging to class 1 (feature)
        n_2 = np.sum(Y == 2) #number of pixels belonging to class 2 (no feature)

        # subsample
        s_1 = np.random.choice(n_1, int(n_1 * sample_fraction[0]))
        s_2 = np.random.choice(n_2, int(n_2 * sample_fraction[1]))

        sampleX1 = X[Y == 1, :][s_1, :]
        sampleY1 = Y[Y == 1][s_1]

        sampleX2 = X[Y == 2, :][s_2, :]
        sampleY2 = Y[Y == 2][s_2]

        features_list.append(sampleX1)
        features_list.append(sampleX2)
        labels_list.append(sampleY1)
        labels_list.append(sampleY2)

    return np.concatenate(features_list), np.concatenate(labels_list)

# create training datasets
features_file_list=[os.path.join('features', f) for f in os.listdir('features') if f.endswith('FEATURES.npy')]
labels_file_list=[os.path.join('features', f) for f in os.listdir('features') if f.endswith('LABELS.npy')]

X, Y = compile_training_dataset_from_precomputed_features(features_file_list, 
                                                          labels_file_list=labels_file_list,
                                                          sample_fraction=(1.0, 0.03))


##################################################################################################
#train a model


# fit random forest classifier (any other classifier)
clf = RandomForestClassifier(n_estimators=250, n_jobs=-1,
                            max_depth=30, max_samples=0.05)
clf.fit(X, Y)


# # dump the model to a file
# os.mkdir("./models")
# rh.dump_model(clf, './models/RF_demo.joblib')


#######################################################################
#make prediction to a test set
from skimage import future
def predict_segmentor(clf: RandomForestClassifier, 
                      features_test_im: np.ndarray) -> np.ndarray:
    """
    Function to predict on a test image
    
    PARAMETERS
    ----------
    clf : RandomForestClassifier (or any scikit-learn classifier)
        The trained classifier used for prediction.
        
    features_test_im : np.ndarray
        (nrow x nkol x k) array of features representing the test image.
        
    RETURNS
    -------
    np.ndarray
        Array of predicted labels representing the predicted segmentation.
    """
    result = future.predict_segmenter(features_test_im, clf)
    return result

import cv2
#im = imread("test_images/04152412.JPG")
#im = imread("test_images/04152434.JPG")
im = imread("test_images/04152704.JPG")
#im = imread('images/original/04152749.JPG')

# compute features
features = im2features(im, sigma_max = 4)

# predict
predicted_segmentation = predict_segmentor(clf, features)
    # - 1 = feature
    # - 2 = no-feature
    # - 0 = buffer (unknown zone near features)

# Make an overlay for plotting
overlay = im.copy()
overlay[predicted_segmentation == 1] = [255, 0, 0]  # paint mask pixels red
alpha = 0.5  # transparency: 0 = only original, 1 = only mask
blended = cv2.addWeighted(im, 1 - alpha, overlay, alpha, 0)

fig = plt.figure(figsize=(12, 4), dpi=250)
plt.imshow(blended)
plt.axis("off")
plt.show()




# # clean detected roots
# roots = clean_predicted_roots(predicted_segmentation, small_objects_threshold=150, closing_diameter = 4)