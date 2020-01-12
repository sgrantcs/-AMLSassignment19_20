# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 19:08:00 2020
Copy of the notebook with the same name with .ipynb extension

@author: sgrant
"""

import cv2
import dlib
import os
import csv
import numpy as np
import random
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
labels_filename = 'labels.csv'
correct_labels = dict()

filelist = os.listdir(images_dir)
image_filelist = []
for file in filelist:
    if file.lower().endswith('.png'):
        image_filelist.append(file)

#read the labels file and write the data into the new labels list called 'correct_labels'
with open('../Datasets/cartoon_set/labels.csv', 'r') as file:
    reader = csv.reader(file, delimiter = '\t')
    next(reader)
    for row in reader:
        correct_labels[row[3]] = row
    #print(correct_labels)
    
#Use Hough Transform to find eye circles in an image
def detect_eye(image_file):
    # Read image as gray-scale
    img = cv2.imread(image_file, cv2.IMREAD_COLOR)
    # Convert to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #test
    # Blur the image to reduce noise
    img_blur = cv2.medianBlur(gray, 5)
    # Apply hough transform on the image
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/2, param1=200, param2=10, minRadius=15, maxRadius=20)
    # Draw detected circles
    #print(circles)
    detected_eyes = circles
    eye = None
    if circles is not None: 
        # Extract eyes
            for detected_eye in detected_eyes[0]: #get coordinates and size of a circle containing an eye
                #print(f"eyes found in file: {image_file}") 
                #print(detected_eye)
                x=int(detected_eye[0])
                y=int(detected_eye[1])
                r=int(detected_eye[2])
                eye = img[y:y+r, x:x+r] #Cut the frame to size in a color image
    return eye

def get_files(image_files): #Define function to get file list, randomly shuffle it and split 80/20
    random.shuffle(image_files)
    training = image_files[:int(len(image_files)*0.8)] #get first 80% of file list
    prediction = image_files[-int(len(image_files)*0.2):] #get last 20% of file list
    return training, prediction

#get training data based on the eye frame converted into HSV
training, prediction = get_files(image_filelist)

training_data = []
training_labels = []
prediction_items = []
prediction_data = []
prediction_labels = []

for item in training:
    image = detect_eye(os.path.join(images_dir, item))
    if image is not None:
        HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        resized_image = cv2.resize(HSV_image, (20, 20))
        reshaped_image = np.ndarray.flatten(resized_image)
        #print('this is a test')
        #print(resized_image)
        #print(reshaped_image)
        training_data.append(reshaped_image) 
        training_labels.append(int(correct_labels[item][1])) #append correct label to the training labels list
print("size of detected eye training set is:", len(training_labels), "images")

for item in prediction:
    image = detect_eye(os.path.join(images_dir, item)) #open image and detect eye in the image
    if image is not None:
        HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        resized_image = cv2.resize(HSV_image, (20, 20))
        reshaped_image =np.ndarray.flatten(resized_image)
        prediction_items.append(item)
        prediction_data.append(reshaped_image) #append image array to training data list
        prediction_labels.append(int(correct_labels[item][1]))
         
print("size of detected eye prediction set is:", len(prediction_data), "images")

classifier = svm.SVC(kernel='linear', C = 1.0)
#print(training_data)
#reshaped_trdata = np.reshape(training_data,(len(training_data),-1))
#print(training_data[0])
classifier.fit(training_data, training_labels)
pred = classifier.predict(prediction_data)


print("Accuracy:", accuracy_score(prediction_labels, pred))