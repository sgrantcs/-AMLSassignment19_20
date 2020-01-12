# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 21:15:01 2020
Copy of the notebook with the same name with .ipynb extension

@author: svetl
"""
import cv2
import dlib
import os
import csv
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from keras.preprocessing import image

# PATH TO ALL IMAGES
#basedir, image_paths
basedir = '../Datasets/cartoon_set'
data_dir = os.environ['BASEDIR']
if data_dir != None: 
    basedir = data_dir
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# The description of the face detector is quoted based on the information provided in the AMLS Lab 2, SVM classification code.
#The code in this model has been modified based on the code provided in the AMLS Lab 2, SVM classification.
#The frontal human faces is found in an image using 68 landmarks.  These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.
# The face detector used here is made using the classic Histogram of Oriented 
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by using dlib's libraries. 

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them 
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)
    
    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image
    print("finished detecting faces")

def extract_features_labels():
    """
    This function extracts the landmarks features for all images in the folder 'Datasets/cartoon_set'.
    It also extracts the shape label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        shape_labels:      an array containing 5 face shape labels (0,1,2,3,4) for each image in
                            which a face was detected
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    shape_labels = dict()
    for line in lines[1:]:
        columns = line.split('\t')
        shape_labels[columns[0]] = int(columns[2])
    
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for img_path in image_paths:
            #print(image_path)
            file_name = img_path.split('.')[2].split('\\')[-1]
            #print(file_name)

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
                        
            features, _ = run_dlib_shape(img)
            
            if features is not None:
                all_features.append(features)
                all_labels.append(shape_labels[file_name])

    landmark_features = np.array(all_features)
    shape_labels = (np.array(all_labels)) # keeps the 5 face shape labels, 0, 1, 2, 3 and 4
    return landmark_features, shape_labels
def get_data():

    X, Y = extract_features_labels()
    Y = np.array([Y, -(Y - 1)]).T
    # Split into a training set and a test set using a stratified k fold
    # Split into a training (75%) and testing (25%) set
   
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    return X_train, Y_train, X_test, Y_test

# sklearn functions implementation
def img_SVM(training_images, training_labels, test_images, test_labels):
    #classifier = svm.LinearSVC(C = 1.0, max_iter=100000) 
    classifier = svm.SVC(kernel='linear', C = 1.0)
    classifier.fit(training_images, training_labels)
    pred = classifier.predict(test_images)
    print("Accuracy:", accuracy_score(test_labels, pred))

    #print(pred)
    return pred

X_train, Y_train, X_test, Y_test = get_data()
num_X_test = len(X_test)
print(f'num_X_test{num_X_test}')
num_X_train = len(X_train)
print(f'num_X_train{num_X_train}')
pred = img_SVM(X_train.reshape((num_X_train, 68*2)), list(zip(*Y_train))[0], X_test.reshape((num_X_test, 68*2)), list(zip(*Y_test))[0])