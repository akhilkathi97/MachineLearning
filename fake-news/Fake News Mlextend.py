# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 23:10:42 2020

@author: akhil
"""

#Load Data
import pandas as pd

train = pd.read_csv('train.csv')
train.dropna()
X = train.drop('label',axis=1)
y =train['label']

#Text Cleaning
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
import re
from spellchecker import SpellChecker  
lookups = Lookups()
lookups.add_table("lemma_rules", {"noun": [["s", ""]]})
lemmatizer = Lemmatizer(lookups)

nlp = spacy.load('en')
spell_check = SpellChecker() 
def text_preprocessing(text):
    text = str(text).lower()
    spell_check_words = [spell_check.correction(re.sub(r'[^\w\s]','',word)) for word in str(text).split(" ")  ]
    lemma_sent = [lemmatizer.lookup(word) for word in spell_check_words if word not in STOP_WORDS]
    cleaned_sent = (' ').join(lemma_sent)
    print(cleaned_sent)
    return cleaned_sent

X['text'] = X['text'].apply(lambda row : text_preprocessing(row))
X['title'] = X['title'].apply(lambda row : text_preprocessing(row))
from sklearn.model_selection import StratifiedKFold


kfold = StratifiedKFold(n_splits=2,shuffle=False,random_state = 42)
for train_index,test_index in kfold.split(X,y):
    X_train,X_test = X.iloc[[train_index]],X.iloc[[test_index]]
    y_train,y_test = y.iloc[[train_index]],y.iloc[[test_index]]

from sklearn.feature_selection import TfidfVectorizer

tfidf = TfidfVectorizer(min_df=0.15,max_df=0.95,norm='l1',binary=True)
tfidf_text = tfidf.fit_transform(X_train['text'])

#Passive Aggressive Classifier
from sklearn.linear_model import PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(C=1.0,validation_fraction = 0.3,early_stopping=True,loss='squared_hinge')
pac.fit(tfidf_text,y)
pac.predict(X_test['text'])

#Multinomial Naive Bayes with TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB(alpha=0.8)
mnb.fit(tfidf_text,y)
mnb.predict(X_test['text'])
 
#Logistic Regression with TfidfVectorizer
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l1',solver='liblinear')
lr.fit(tfidf_text,y)
lr.predict(X_test['text'])

from mlxtend.classifier import EnsembleVoteClassifier

ens = EnsembleVoteClassifier(clfs=[pac,mnb,lr],voting='soft',weights=[2,1,2],fit_base_estimators=True)
ens.fit(tfidf_text,y)
ens.predict(X_test['text'])