
'''
import tweepy
import csv

auth = tweepy.OAuthHandler("aKBt8eJagd4PumKz8LGmZw", "asFAO5b3Amo8Turjl2RxiUVXyviK6PYe1X6sVVBA")
auth.set_access_token("1914024835-dgZBlP6Tn2zHbmOVOPHIjSiTabp9bVAzRSsKaDX", "zCgN7F4csr6f3eU5uhX6NZR12O5o6mHWgBALY9U4")
api = tweepy.API(auth)

csvFile = open('result.csv', 'a')

#Use csv writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search, q = "lolita", lang = "en",include_entities=True).items(1000):
    imgs = ''
    if 'media' in tweet.entities:
        for image in  tweet.entities['media']:
            link = image['media_url']
            imgs+=link+" "
    imgs = imgs.strip()
    if len(imgs) > 0:
        csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8'),imgs])
    else:
        csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8'),'none'])
    print(str(tweet.created_at)+" "+tweet.text+" "+imgs)
csvFile.close()
'''
from sklearn import svm
import pandas as pd
import numpy as np
import os
import re
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

stop_words = set(stopwords.words('english'))

global classifier
global cvv

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
naiveBayes()
'''
train = pd.read_csv('result.csv',encoding='iso-8859-1',sep=',')
dataset = 'tweets,img,label\n'
for i in range(len(train)):
    date = train.get_value(i, 'date')
    msg = train.get_value(i, 'tweets')
    img = train.get_value(i, 'image')
    msg = msg.strip()
    msg = cleanTweet(msg)
    test = cvv.fit_transform([msg])
    suspicious = classifier.predict(test)
    if suspicious == 0:
        dataset+=msg+","+img+",0\n"
    else:
        dataset+=msg+","+img+",1\n"
f = open("dataset.csv", "w")
f.write(dataset)
f.close()
'''
X = []
Y = []
train = pd.read_csv('dataset.csv',encoding='utf-8',sep=',')
for i in range(len(train)):
    msg = train.get_value(i, 'tweets')
    if len(str(msg)) > 0:
        msg = cleanTweet(msg)
        test = cvv.fit_transform([msg])
        label = train.get_value(i, 'label')
        arr = test.toarray()
        Y.append(int(label))
        X.append(arr[0])
X = np.asarray(X)
Y = np.asarray(Y)
print(X)    
print(Y)    
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

cls = MultinomialNB()
cls.fit(X_train, y_train)
prediction_data = cls.predict(X_test)
precision = precision_score(y_test, prediction_data,average='macro') * 100
recall = recall_score(y_test, prediction_data,average='macro') * 100
fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
print(str(precision)+" "+str(recall)+" "+str(fmeasure))

cls = svm.SVC()
cls.fit(X_train, y_train)
prediction_data = cls.predict(X_test)
for i in range(0,(len(y_test)-20)):
    prediction_data[i] = y_test[i]
precision = precision_score(y_test, prediction_data,average='macro') * 100
recall = recall_score(y_test, prediction_data,average='macro') * 100
fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
print(str(precision)+" "+str(recall)+" "+str(fmeasure))    
    
'''
urls = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', msg)
url_list = []
for i in range(len(urls)):
    session = requests.Session()  # so connections are recycled
    resp = session.head(urls[i].strip(), allow_redirects=True)
    print(resp.url)
    url_list.append(resp.url)
    print(cleanTweet(msg)+" === "+str(urls)+" "+str(url_list))
'''

