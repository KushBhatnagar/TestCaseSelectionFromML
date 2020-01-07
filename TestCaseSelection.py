# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 08:42:49 2020

This script will priortize regression test cases based on machine learning algorithm

@author: Kush
"""

#importing required libraries 
import numpy as np
import pandas as pd
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

#Importing test case dump
dataset=pd.read_csv("TestCases_TestData.csv")

corpus  = []
pstem = PorterStemmer()
for i in range(dataset['TCD'].shape[0]):
    #Remove unwanted words
    text = re.sub("[^a-zA-Z]", ' ', dataset['TCD'][i])
    #Transform words to lowercase
    text = text.lower()
    text = text.split()
    #Remove stopwords then Stemming it
    text = [pstem.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    #Append cleaned tweet to corpus
    corpus.append(text)
    
print("Corpus created successfully") 

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Gausian to the Training set
classifier_gnb = GaussianNB()
classifier_gnb.fit(X_train, y_train)
# Predicting the Test set results
y_pred_gnb = classifier_gnb.predict(X_test)
# Making the Confusion Matrix
cm_gnb = confusion_matrix(y_test, y_pred_gnb)

# Fitting Naive Bayes to the Training set
classifier_nb = MultinomialNB(alpha=0.1)
classifier_nb.fit(X_train, y_train)
# Predicting the Test set results
y_pred_nb = classifier_gnb.predict(X_test)
# Making the Confusion Matrix
cm_nb = confusion_matrix(y_test, y_pred_nb)

# Fitting K - Nearest Neighbors Model to the Training set
classifier_knn = KNeighborsClassifier(n_neighbors = 7,
                                       weights = 'distance',
                                      algorithm = 'brute')
classifier_knn.fit(X_train, y_train)
# Predicting the Test set results
y_pred_knn = classifier_knn.predict(X_test)
# Making the Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

# Fitting Gradient Boosting Model to the Training set
classifier_gb = GradientBoostingClassifier(loss = 'deviance',
                                                   learning_rate = 0.01,
                                                   n_estimators = 100,
                                                   max_depth = 30,
                                                   random_state=55)
classifier_gb.fit(X_train, y_train)
# Predicting the Test set results
y_pred_gb = classifier_gb.predict(X_test)
# Making the Confusion Matrix
cm_gb = confusion_matrix(y_test, y_pred_gb)

# Fitting Decision Tree Model to the Training set
classifier_dt = DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = None, 
                                           splitter='best', 
                                           random_state=55)
classifier_dt.fit(X_train, y_train)
# Predicting the Test set results
y_pred_dt = classifier_dt.predict(X_test)
# Making the Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)




