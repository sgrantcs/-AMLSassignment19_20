# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 19:08:48 2020
Copy of the notebook with the same name with .ipynb extension

@author: sgrant
"""
import cv2
import os
import numpy as np
import dlib
import csv
import random
 
#create basedir, image_paths, create labels file
basedir = '../Datasets/celeba'
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'
correct_labels = dict()

#read the labels file and write the data into the new labels list called 'correct_labels'
with open('../Datasets/celeba/labels.csv', 'r') as file:
    reader = csv.reader(file, delimiter = '\t')
    next(reader)
    for row in reader:
        correct_labels[row[1]] = row
    #print(correct_labels)

filelist = os.listdir(images_dir)
image_filelist = []
for file in filelist:
    if file.lower().endswith('.jpg'):
        image_filelist.append(file)

face_detectors = [ cv2.CascadeClassifier("haarcascade_frontalface_default.xml"),
                  cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml"),
                  cv2.CascadeClassifier("haarcascade_frontalface_alt.xml"),
                  cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml"), 
                  cv2.CascadeClassifier("haarcascade_profileface.xml") ]

def detect_face(image_file):

    frame = cv2.imread(image_file) #Open image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #Normalise histogram
    
    #Detect face by going through different classifiers. 
    detected_face = []
    
    for face_detector in face_detectors:
        
        face = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        
        if ( len(face) == 1 ):
            
            detected_face = face 
            break 

    face_image = [] 
    
    if len(detected_face) >0:

        # Extract face
        for (x, y, w, h) in detected_face: #get coordinates and size of rectangle containing face
            #print(f"face found in file: {image_file}") 
            
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            clahe_image = clahe.apply(gray)
            try:
                face_image = cv2.resize(clahe_image, (178, 218)) #Resize face so all images have same size
            except:
                pass #If error, pass file

    return face_image 

def get_files(image_files): #Define function to get file list, randomly shuffle it and split 80/20
    random.shuffle(image_files)
    training = image_files[:int(len(image_files)*0.8)] #get first 80% of file list
    prediction = image_files[-int(len(image_files)*0.2):] #get last 20% of file list
    return training, prediction

#fishface = cv2.face.FisherFaceRecognizer_create() #Initialize Fisher face classifier
fishface = cv2.face.LBPHFaceRecognizer_create() # Initialise alternative face classifier

training, prediction = get_files(image_filelist)

training_data = []
training_labels = []
prediction_data = []
prediction_labels = []

print("size of initial training set is:", len(training), "images")

for item in training:
    image = detect_face(os.path.join(images_dir, item)) #open image
    if len(image) >0:
        training_data.append(image) #append image array to training data list
        training_labels.append(int(correct_labels[item][3])) #append correct label to the training labels list
print("size of detected training set is:", len(training_labels), "images")

fishface.train(training_data, np.array(training_labels)) #use openCV train function to train the model
print("the training has finished")

for item in prediction:
    image = detect_face(os.path.join(images_dir, item)) #open image and detect face in the image
    if len(image) >0:
        prediction_data.append((item, image)) #append image array to training data list
         
print("size of detected predicted set is:", len(prediction_data), "images")

correct = 0

for item, image in prediction_data:
        pred, conf = fishface.predict(image) #use openCV predict function to predict the face emotion
        prediction_labels.append(pred) #append predicted label to the prediction_labels list
        if pred == int(correct_labels[item][3]): #compare prediction value with correct value. 
            correct += 1 #If correct match, increment correct counter.

            #create smiles_output file for prediction analysis:
with open('../Datasets/celeba/smiles_output.csv', 'w', newline='') as outfile: 
    writer = csv.writer(outfile, delimiter = ',')
    writer.writerow(['img_name', 'correct_smiling', 'predicted_smiling'])
    label_index = 0
    for item, image in prediction_data:
        correct_labels_row = correct_labels[item]
        new_prediction = prediction_labels[label_index]
        writer.writerow([item, correct_labels_row[3], new_prediction])
        label_index +=1
            
model_accuracy = (100*correct)/len(prediction_data)
print("Model accuracy : " + str(model_accuracy) + "%")