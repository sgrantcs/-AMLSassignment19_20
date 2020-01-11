# -*- coding: utf-8 -*-
"""
File .py version created on Sat Jan 11 18:48:42 2020
Copy of .ipynb file with the same name.

@author: sgrant
"""

import os
import cv2
import csv

#basedir, image_paths and labels name and list for label information
basedir = '../Datasets/celeba'
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'
correct_labels = dict()

with open('../Datasets/celeba/labels.csv', 'r') as file:
    reader = csv.reader(file, delimiter = '\t')
    next(reader)
    for row in reader:
        correct_labels[row[1]] = row

#Create a list for gender labels
gender_list = ['1', '-1']

#load caffe model and model_mean_values
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

#Load the pre-existing Haar Cascade model for facial detection. 
#Five different models are loaded to improve the face recognition.
face_detectors = [ cv2.CascadeClassifier("haarcascade_frontalface_default.xml"),
                  cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml"),
                  cv2.CascadeClassifier("haarcascade_frontalface_alt.xml"),
                  cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml"), 
                  cv2.CascadeClassifier("haarcascade_profileface.xml") ]
#Process image and identify faces
image_filelist = os.listdir(images_dir)
counter=0
predicted_labels = list()
for file in image_filelist:
    if file.lower().endswith('.jpg'):
        image_path = os.path.join(images_dir, file)
        img = cv2.imread(image_path,1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for face_detector in face_detectors: 
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            if len(faces) >0:
                break
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face_img = img[y:y+h, h:h+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        #Predict Gender using gender_net 
        gender_net.setInput(blob)
        gender_preds = gender_net.forward() #predict gender
        gender = gender_list[gender_preds[0].argmax()] #create gender labels list
        predicted_labels.append((file, gender))
        counter += 1
        #print(gender)

#create a new labels output file for correct labels and new (predicted) labels
with open('../Datasets/celeba/output.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter = ',')
    writer.writerow(['img_name', 'correct_gender', 'predicted_gender']) #write column names for new file
    for predicted_label in predicted_labels:
        correct_labels_row = correct_labels[predicted_label[0]] #copy correct labels from labels file
        writer.writerow([predicted_label[0], correct_labels_row[2], predicted_label[1]])
        
#To determine model accuracy, the prediction output is compared with the existing labels; 
#if predictions match, count correct predictions
#model accuracy is calculated as % of total number of correct classifications.
accurate_score = 0
for predicted_label in predicted_labels:
    correct_labels_row = correct_labels[predicted_label[0]] 
    if predicted_label[1] == correct_labels_row[2]: #compare predicted labels with correct labels
        accurate_score +=1 #increment correct score
model_accuracy = (accurate_score*100)/len(predicted_labels) #calculate model accuracy
print("Model accuracy : " + str(model_accuracy) + "%") #print model accuracy value
#print(len(predicted_labels))