# import libraries
import cv2
import os
import csv
import numpy as np
import dlib
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,accuracy_score
from sklearn.svm import SVC
from keras.preprocessing import image
import pandas as pd
from sklearn.datasets import load_iris

# ======================================================================================================================
# Data preprocessing


#data_train, data_val, data_test = data_preprocessing(args...)
# ======================================================================================================================
# Task A1 - gender recognition
#The A1.A1_gender_detectio_Fisherface model uses the Fisherface classifier
os.environ['BASEDIR'] = './Datasets/celeba'
import A1.A1_gender_detection_Fisherface
#The A1.A1_gender_detection uses the pre-trained CNN model
import A1.A1_gender_detection

# all output and accuracy calculations are printed in the task code.
#The accuracy of the test model achieved is 


# ======================================================================================================================
# Task A2 -
os.environ['BASEDIR'] = './Datasets/celeba'
import A2.A2_smile_detection_model
 
# all output and accuracy calculations are printed in the task code.

# ======================================================================================================================
# Task B1
os.environ['BASEDIR'] = './Datasets/cartoon_set'
import B1.B1_SVM_classification

# all output and accuracy calculations are printed in the task code.
#Note that the model training and predictions for B1 model take a long time (over an hour on a home laptop).

# ======================================================================================================================
# Task B2
os.environ['BASEDIR'] = './Datasets/cartoon_set'
import B2.B2_EyeColor_Model

# all output and accuracy calculations are printed in the task code.
#Note that the model training and predictions for B2 model take a long time (over an hour on a home laptop).

# ======================================================================================================================
## Print out your results with following format:
#print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
#                                                        acc_A2_train, acc_A2_test,
#                                                        acc_B1_train, acc_B1_test,
#                                                        acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
acc_A1_train = 'Calculated in the task code'
acc_A1_test = 'Calculated in the task code'
acc_A2_train = 'Calculated in the task code'
acc_A2_test = 'Calculated in the task code'
acc_B1_train = 'Calculated in the task code'
acc_B1_test = 'Calculated in the task code'
acc_B2_train = 'Calculated in the task code'
acc_B2_test = 'Calculated in the task code'