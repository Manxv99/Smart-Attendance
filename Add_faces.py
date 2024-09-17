import cv2  #OpenCV (Open Source Computer Vision) library used for computer vision tasks. It provides functionalities for image processing, video capturing, facial recognition, object detection, and more.
import pickle  #pickle is a Python module used to serialize (convert to a byte stream) and deserialize (reconstruct) Python objects. It is useful for saving and loading model data, configuration settings, or any Python object to a file
import numpy as np
import os  #os provides a way to interact with the operating system
import time  #time is a standard Python library that provides various time-related functions.
from datetime import datetime  #datetime is a module for handling date and time in Python. Importing datetime from the module provides classes for manipulating dates and times.

#open a video camera object using the default inbuilt camera(0)
video = cv2.VideoCapture(0)

#load the Haar Cascade Classifier for face detection
facedetect = cv2.CascadeClassifier("Data/haarcascade_frontalface_default.xml")
#cv2.CascadeClassifier:
# This is a class from the OpenCV (cv2) library used for object detection.
# It loads a pre-trained Haar Cascade classifier from an external file, which contains the data required for detecting objects (like faces).
# The file "haarcascade_frontalface_default.xml" is one of the pre-trained models provided by OpenCV. It includes data about different patterns that represent features of human faces (like the eyes, nose, and mouth).
# When the classifier is loaded, OpenCV can use it to detect faces in images or video frames.

#initialize empty list to store face data
faces_data = []

#counter to keep track of no. of frames processed
i=0

#get user input for their name
name = input("Enter your name: ")

#loop to capture video frames and detect faces
while True:
    #capture a frame from the video
    ret, frame = video.read()

    #convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces in the frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5) #the function detectMultiScale in your code is used to detect faces in the grayscale image gray, scaling the image down by 1.3 for each pass and requiring each potential face to be confirmed by at least 5 detections. The output, faces, will be a list of rectangles (x, y, width, height) where faces were detected.

    #iterate over detected faces
    for(x, y, w, h) in faces:
        #crop face region from the frame
        crop_img = frame[y:y+h, x:x+w, :]

        #resize the cropped face image to 50x50 pixels
        resized_img = cv2.resize(crop_img, (50,50))

        #append the resized face image to face_data list every 5 frames
        if len(faces_data)<=5 and i%5==0:
            faces_data.append(resized_img)

        i = i+1

        #display count of captured faces on the frame
        #frame: The image or video frame on which the text will be drawn.
# str(len(faces_data)): The text to be displayed. Here, it converts the length of the faces_data list (which presumably contains detected faces) to a string. This displays the number of detected faces on the frame.
# (50, 50): The bottom-left corner of the text string in the image. In this case, the text will start 50 pixels from the left and 50 pixels from the top of the frame.
# 1: The font scale factor that affects the size of the font. The value 1 indicates that the base font size should be used.
# (50, 50, 225): The color of the font in BGR (Blue, Green, Red) format. This specific combination will produce a reddish color for the text because the Red value is high.
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 225))

        #draw rectangle around the detected face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

    #display current frames with annotations
    cv2.imshow("Frame", frame)

    #wait for a key press or until 5 faces are captured
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 5:
        break

#release the video capture and close all windows
video.release()
cv2.destroyAllWindows()

#convert the list of faces images to a numpy array and reshape it
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(5,-1)  #This line reshapes the array faces_data into a new shape that has 5 rows and a number of columns calculated automatically to fit all the original data. The -1 tells NumPy to calculate the necessary number of columns based on the length of the array and the other dimension specified (5 rows in this case).

#check if 'names.pkl' is present in 'Data/' directory
if 'names.pkl' not in os.listdir('Data/'):
    #create a list with the entered name repeated 5 times
    names = [name]*5
    #save list to 'names.pkl
    with open('Data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

else:
    with open('Data/names.pkl', 'rb') as f:
        names = pickle.load(f)

    #append the entered name 5 times to the existing list
    names = names + [name]*5
    #save the updated list to 'names.pkl'
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

#check if 'faces_data.pkl' is present in 'Data/' directory
if 'faces_data.pkl' not in os.listdir('Data/'):
    #save the numpy array faces_data to faces_data.pkl
    with open('Data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)

else:
    #load existing array
    with open('Data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)

    #append the new array faces_data to existing array
    faces = np.append(faces, faces_data, axis=0)

    #save updates array to .pkl
    with open('Data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)

