# README

AMLS Term 1 Project description: 
### Libraries
The following libraries were used for the four tasks: 
os, cv2, csv, dlib, NumPy 
OS - for file manipulation 
cv2 - for image operations and running DNN model, and CSV for labels file manipulation
dlib - for image processing, 
NumPy - for working with image arrays
random - for randomly splitting datasets into training and test sets
sklearn.model_selection - for train_test_split function 
The SVM was implemented using Scikit-learn library. sklearn library was used to import svm, 
sklearn.svm was used to download SVC model (Linear kernel was used).

### Other required files 
Other requirements for running the models:
Open framework caffemodel was used for gender detection CNN. 
The caffe model and the Prototxt file can be downloaded from Tal Hassner's web site: https://talhassner.github.io/home/publication/2015_CVPR
Haar Cascades were used for face detection; these are availabel for download from OpenCV Docs: <https://docs.opencv.org/master/>

2. AMLS Project File Organisation: 
$ -- AMLS_19-20_SN19132626
main.py
README.md (this file)
|--A1
|--|--A1_gender_detection.py 
|--|--A1_gender_detection.ipynb (Jupyter notebook with the code and commentary)
|--|--deploy_gender.prototxt (source: https://talhassner.github.io/home/publication/2015_CVPR)
|--|--gender_net.caffemodel (Model source at Tal Hassner's web site: https://drive.google.com/open?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ)
|--|--haarcascade_frontalface_alt.xml
|--|--haarcascade_frontalface_alt2.xml
|--|--haarcascade_frontalface_alt_tree.xml
|--|--haarcascade_frontalface_default.xml
|--|--haarcascade_profileface.xml
|--A2
|--|--A2_smile_detection_model.py
|--|--A2_smile_detection_model.ipynb
|--|--haarcascade_eye.xml
|--|--haarcascade_frontalface_alt.xml
|--|--haarcascade_frontalface_alt2.xml
|--|--haarcascade_frontalface_alt_tree.xml
|--|--haarcascade_frontalface_default.xml
|--|--haarcascade_profileface.xml
|--B1
|--|--B1_SVM_classification.ipynb
|--|--B1_SVM_classification.py
|--|--shape_predictor_68_face_landmarks.dat
|--B2
|--|--B2_EyeColor_model.ipynb
|--|--B2_EyeColor_model.py
|--Datasets
|--|--cartoon_set
|--|--|--img[10,000 images] 
|--|--|--labels.csv
|--|--celeba
|--|--|--img[5,000 images]
|--|--|--labels.csv


