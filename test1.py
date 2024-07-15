import os
import numpy as np
import cv2
import pandas as pd
import shutil
import requests

def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('model/deploy_age.prototxt', 'model/age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('model/deploy_gender.prototxt', 'model/gender_net.caffemodel')
    return(age_net, gender_net)

def download(url):
    response = requests.get(url, stream=True)
    with open('img.png', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    out_file.close()
    del response

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_net, gender_net = load_caffe_models()
age_list=['2','6','12','20','32','43','53','100']
gender_list=['Male','Female']

train = pd.read_csv('dataset.csv',encoding='utf-8',sep=',')
for i in range(len(train)):
    msg = train.get_value(i, 'img')
    label = train.get_value(i, 'label')
    print(str(msg)+" "+str(label))
    if msg != 'none':
        download(msg)
        example_image = 'img.png'
        face_img = cv2.imread(example_image)
        temp = cv2.imread(example_image)
        frame = cv2.imread(example_image,0)
        faces = face_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        #faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if(len(faces)>0):
            print("Found {} faces".format(str(len(faces))))
            for (x, y, w, h )in faces:
                cv2.rectangle(temp, (x, y), (x+w, y+h), (255, 255, 0), 2)
                face_img = temp[y:y+h, h:h+w].copy()
                blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                print("Gender : " + gender)
                blob = cv2.dnn.blobFromImage(temp, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]
                age = int(age)
                age_msg = ''
                if age > 14:
                    age_msg = 'Over 14 years old'
                else:
                    age_msg = "Under 14 years old"
                print("Age Range: " + age_msg)
                cv2.putText(temp, str(gender)+" "+str(age_msg), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("ll",temp)
            cv2.waitKey(0)    
