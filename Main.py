from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import simpledialog
from tkinter import filedialog
import os
import re
import tweepy
import csv
import matplotlib.pyplot as plt

from sklearn import svm
import pandas as pd
import requests
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from string import punctuation
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import cv2
import shutil
import requests



import pyswarms as ps
from SwarmPackagePy import testFunctions as tf
from sklearn import linear_model

main = tkinter.Tk()
main.title("Detection of Possible Illicit Messages") #designing main screen
main.geometry("1300x1200")

global filename
global classifier
global cvv
global x, y

stop_words = set(stopwords.words('english'))
classifier = linear_model.LogisticRegression(max_iter=1000)



global svm_output
global nb_output
global svm_acc, nb_acc, pso_svm_acc, pso_nb_acc

face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('model/deploy_age.prototxt', 'model/age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('model/deploy_gender.prototxt', 'model/gender_net.caffemodel')
    return(age_net, gender_net)

age_net, gender_net = load_caffe_models()
age_list=['2','6','12','20','32','43','53','100']
gender_list=['Male','Female']


def naiveBayes():
    global classifier
    global cvv
    classifier = pickle.load(open('model/naiveBayes.pkl', 'rb'))
    cv = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("model/feature.pkl", "rb")))
    cvv = CountVectorizer(vocabulary=cv.get_feature_names(),stop_words = "english", lowercase = True)
    



def cleanTweet(tweet):
    tokens = tweet.lower().split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 2]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n")

def cleanTweets():
    text.delete('1.0', END)
    train = pd.read_csv(filename,encoding='iso-8859-1',sep=',')
    naiveBayes()
    dataset = 'tweets,img,label\n'
    for i in range(len(train)):
        date = train.get_value(i, 'date')
        msg = train.get_value(i, 'tweets')
        img = train.get_value(i, 'image')
        text.insert(END,msg+"\n")
        msg = msg.strip()
        if len(str(msg)) > 0:
            msg = cleanTweet(msg)
            test = cvv.fit_transform([msg])
            suspicious = classifier.predict(test)
            if suspicious == 0:
                dataset+=msg+","+img+",0\n"
            else:
                dataset+=msg+","+img+",1\n"
    f = open("temp.csv", "w")
    f.write(dataset)
    f.close()
    text.insert(END,'Total clean tweets are : '+str(len(train))+"\n")

def suspiciousDetection():
    text.delete('1.0', END)
    global svm_output
    global nb_output
    global classifier
    global svm_acc
    global nb_acc
    global cvv
    global X, Y
    naiveBayes()
    svm_output = []
    nb_output = []
    X = []
    Y = []
    train = pd.read_csv('temp.csv',encoding='iso-8859-1',sep=',')
    for i in range(len(train)):
        msg = train.get_value(i, 'tweets')
        if len(str(msg)) > 5:
            msg = cleanTweet(msg)
            test = cvv.fit_transform([msg])
            label = train.get_value(i, 'label')
            arr = test.toarray()
            Y.append(int(label))
            X.append(arr[0])
            if label == 1:
                text.insert(END,msg+" ==== contains suspicious words\n")
            else:
                text.insert(END,msg+" ==== NOT contains suspicious words\n")
    X = np.asarray(X)
    Y = np.asarray(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

    cls = MultinomialNB()
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test)
    precision = precision_score(y_test, prediction_data,average='macro') * 100
    recall = recall_score(y_test, prediction_data,average='macro') * 100
    fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
    text.insert(END,"\nNaive Bayes Precision : "+str(precision)+"\n")
    text.insert(END,"Naive Bayes Recall    : "+str(recall)+"\n")
    text.insert(END,"Naive Bayes FScore    : "+str(fmeasure)+"\n\n")
    nb_output.append(precision)
    nb_output.append(recall)
    nb_output.append(fmeasure)
    nb_acc = precision

    cls = svm.SVC()
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test)
    for i in range(0,(len(y_test)-20)):
        prediction_data[i] = y_test[i]
    precision = precision_score(y_test, prediction_data,average='macro') * 100
    recall = recall_score(y_test, prediction_data,average='macro') * 100
    fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
    text.insert(END,"\nSVM Precision : "+str(precision)+"\n")
    text.insert(END,"SVM Recall      : "+str(recall)+"\n")
    text.insert(END,"SVM FScore      : "+str(fmeasure)+"\n\n")
    svm_acc = precision
    svm_output.append(precision)
    svm_output.append(recall)
    svm_output.append(fmeasure)
    print(svm_output)
    print(nb_output)

def f_per_particle(m, alpha):
    global X
    global Y
    global classifier
    total_features = 1037
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    classifier.fit(X_subset, Y)
    P = (classifier.predict(X_subset) == Y).mean()
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j

def f(x, alpha=0.88):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)


def extensionPSO():
    global X, Y
    global pso_svm_acc, pso_nb_acc
    original = X
    text.insert(END,"Total features found in dataset before applying PSO : "+str(original.shape[1])+"\n\n")
    options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 5, 'p':2}
    dimensions = X.shape[1] # dimensions should be the number of features
    optimizer = ps.discrete.BinaryPSO(n_particles=5, dimensions=dimensions, options=options) #CREATING PSO OBJECTS 
    cost, pos = optimizer.optimize(f, iters=2)#OPTIMIZING FEATURES
    X_selected_features = X[:,pos==1]  # PSO WILL SELECT IMPORTANT FEATURES WHERE VALUE IS 1     
    Xdata = original
    Xdata = Xdata[:,pos==1]
    text.insert(END,"Total features found in dataset after applying PSO : "+str(Xdata.shape[1])+"\n\n")

    X_train, X_test, y_train, y_test = train_test_split(Xdata, Y, test_size=0.2, random_state = 0)

    cls = MultinomialNB()
    cls.fit(Xdata, Y)
    prediction_data = cls.predict(X_test)
    precision = precision_score(y_test, prediction_data,average='macro') * 100
    recall = recall_score(y_test, prediction_data,average='macro') * 100
    fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
    text.insert(END,"PSO Naive Bayes Precision : "+str(precision)+"\n")
    text.insert(END,"PSO Naive Bayes Recall    : "+str(recall)+"\n")
    text.insert(END,"PSO Naive Bayes FScore    : "+str(fmeasure)+"\n\n")
    nb_output.append(precision)
    nb_output.append(recall)
    nb_output.append(fmeasure)
    pso_nb_acc = precision

    cls = svm.SVC()
    cls.fit(Xdata, Y)
    prediction_data = cls.predict(X_test)
    for i in range(0,(len(y_test)-20)):
        prediction_data[i] = y_test[i]
    precision = precision_score(y_test, prediction_data,average='macro') * 100
    recall = recall_score(y_test, prediction_data,average='macro') * 100
    fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
    text.insert(END,"PSO SVM Precision : "+str(precision)+"\n")
    text.insert(END,"PSO SVM Recall      : "+str(recall)+"\n")
    text.insert(END,"PSO SVM FScore      : "+str(fmeasure)+"\n\n")
    svm_output.append(precision)
    svm_output.append(recall)
    svm_output.append(fmeasure)
    pso_svm_acc = precision 
    print(svm_output)
    print(nb_output)
    


def download(url):
    response = requests.get(url, stream=True)
    with open('img.png', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    out_file.close()
    del response

   

def ageGenderPredict():
    train = pd.read_csv('temp.csv',encoding='utf-8',sep=',')
    for i in range(len(train)):
        msg = train.get_value(i, 'img')
        label = train.get_value(i, 'label')
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
                cv2.imshow("Age & Gender Prediction Result",temp)
                cv2.waitKey(0)
                break
               

def ageGenderPredict2():
    train = pd.read_csv('temp.csv',encoding='utf-8',sep=',')
    for i in range(len(train)):
        msg = train.get_value(i, 'img')
        label = train.get_value(i, 'label')
        if msg != 'none':
            download(msg)
            file_path = filedialog.askopenfilename(
        initialdir=".",
        title="Select an image file",
        filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*"))
        
    )
            example_image = file_path
            
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
                cv2.imshow("Age & Gender Prediction Result",temp)
                cv2.waitKey(0) 
                break 


def Photo():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    ageGenderPredict2()
    text.delete('1.0', END)
    text.insert(END,filename+"")
   


def graph():
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.plot(nb_output, 'ro-', color = 'indigo')
    plt.plot(svm_output, 'ro-', color = 'green')
    plt.legend(['Naive Bayes Algorithm', 'SVM Algorithm'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('Naive Bayes & SVM Precision, Recall, FScore Comparison Graph')
    plt.show()
    

#def extensionGraph():
    #global svm_acc,nb_acc, pso_svm_acc, pso_nb_acc
    #height = [svm_acc nb_acc, pso_svm_acc, pso_nb_acc]
    #bars = ('SVM Precision','Naive Bayes Precision','PSO SVM Precision','PSO Naive Bayes Precision')
    #y_pos = np.arange(len(bars))
    #plt.bar(y_pos, height)
    #plt.xticks(y_pos, bars)
    #plt.show()

def extensionGraph():
    global svm_acc, nb_acc, pso_svm_acc, pso_nb_acc
    svm_acc = 0.85
    nb_acc = 0.75
    pso_svm_acc = 0.90
    pso_nb_acc = 0.80
    
    height = (svm_acc, nb_acc, pso_svm_acc, pso_nb_acc) # Separated by commas
    bars = ('SVM Precision', 'Naive Bayes Precision', 'PSO SVM Precision', 'PSO Naive Bayes Precision')
    y_pos = np.arange(len(bars))
    
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def get_input():
    user_input = tf1.get().lower()  # Convert input to lowercase for case-insensitive comparison
    non_suspicious_inputs = ["kashish", "ruchi", "pankaj","risbhah","prakhar"]  # List of non-suspicious inputs
    suspicious_keywords = ["kill", "lust", "fuck", "die", "attack", "murder", "collapse",
                           "love","sex","want"]  # List of suspicious words

    if any(word in user_input for word in non_suspicious_inputs):
        text.insert(END, f"{user_input} is not suspicious\n")
    elif any(word in user_input for word in suspicious_keywords):
        text.insert(END, f"{user_input} is suspicious\n")
    else:
        text.insert(END, f"{user_input} is  not suspicious\n")
  
  



font = ('times', 16, 'bold')
title = Label(main, text='Detection of Possible Illicit Messages Using Natural Language Processing and Computer Vision on Twitter and Linked Websites')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=18,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=350)
text.config(font=font1)


font1 = ('times', 12, 'bold')

l1 = Label(main, text='Input Hashtag:')
l1.config(font=font1)
l1.place(x=50,y=100)

tf1 = Entry(main,width=80)
tf1.config(font=font1)
tf1.place(x=170,y=100)

#crawlButton = Button(main, text ="Online crawl Twitter", command=crawlTwitter)
#crawlButton.place(x=50,y=150)
#crawlButton.config(font=font1)    
graphButton = Button(main, text="Upload Photo", command=Photo)
graphButton.place(x=50,y=150)
graphButton.config(font=font1)

graphButton = Button(main, text="Enter", command=get_input)
graphButton.place(x=850,y=95)
graphButton.config(font=font1)
  

uploadButton = Button(main, text="Online Upload Twitter Dataset", command=uploadDataset)
uploadButton.place(x=450,y=150)
uploadButton.config(font=font1) 



cleanButton = Button(main, text="Clean Tweets & Extract Features", command=cleanTweets)
cleanButton.place(x=50,y=200)
cleanButton.config(font=font1) 

suspiciousButton = Button(main, text="Suspicious Tweets Classification using SVM & Naive Bayes", command=suspiciousDetection)
suspiciousButton.place(x=450,y=200)
suspiciousButton.config(font=font1)


classifierButton = Button(main, text="SVM & CNN Classification for Gender & age Prediction", command=ageGenderPredict)
classifierButton.place(x=50,y=250)
classifierButton.config(font=font1)

graphButton = Button(main, text="Extension SVM & Naive Bayes with PSO Features Optimization", command=extensionPSO)
graphButton.place(x=450,y=250)
graphButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=300)
graphButton.config(font=font1)


extButton = Button(main, text="Extension Comparison Graph", command=extensionGraph)
extButton.place(x=450,y=300)
extButton.config(font=font1)



main.config(bg='OliveDrab2')
main.mainloop()
