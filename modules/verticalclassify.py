import os
from flask import Flask, jsonify, request
import json
import io
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
#nltk.download('punkt') # one time execution
import re
from urllib.request import urlopen
from sklearn.feature_extraction.text import TfidfTransformer

def pre_process(text):
    # lowercase
    text=text.lower()
    #remove tags
    text=re.sub("</?.*?>"," <> ",text)
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    return text

def get_stop_words(stop_file_path):
    """load stop words """
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results


def get_keywords(docs):
#create a vocabulary of words, 
  #ignore words that appear in 85% of documents, 
  #eliminate stop words
  docs = sent_tokenize(docs)
  #print(docs)
  #load a set of stop words
  stopwords = get_stop_words("data/stopwords.txt")
  cv = CountVectorizer(max_df=0.85, stop_words=stopwords)
  word_count_vector=cv.fit_transform(docs)
  word_count_vector.shape
  cv=CountVectorizer(max_df=0.85,stop_words=stopwords,max_features=10000)
  word_count_vector=cv.fit_transform(docs)
  word_count_vector.shape
  tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
  tfidf_transformer.fit(word_count_vector)

  # you only needs to do this once
  feature_names=cv.get_feature_names()
  # get the document that we want to extract keywords from
  doc=' '.join(docs)
  #generate tf-idf for the given document
  tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
  #sort the tf-idf vectors by descending order of scores
  sorted_items=sort_coo(tf_idf_vector.tocoo())
  #extract only the top n; n here is 10
  keywords=extract_topn_from_vector(feature_names,sorted_items,10)
  # now print the results
  #print("\n===Keywords===")
  #for k in keywords:
  #    print(k, keywords[k])
  return keywords


def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)
            #print(path)
            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                lines.append(line)
            f.close()
            message = '\n'.join(lines)
            #print(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

def classifyContent(content):
    example_counts = vectorizer.transform([content])
    predictions = classifier.predict(example_counts)
    predictions

data = DataFrame({'message': [], 'class': []})
data = data.append(dataFrameFromDirectory('data/verticals/kyv', 'Know-Your-Value'))
data = data.append(dataFrameFromDirectory('data/verticals/think', 'Think'))
data = data.append(dataFrameFromDirectory('data/verticals/mach', 'Mach'))
data = data.append(dataFrameFromDirectory('data/verticals/better', 'Better'))

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)

def classify_vertical(post_content):
    example_counts = vectorizer.transform([post_content])
    predictions = classifier.predict(example_counts)
    predction_msg = ''.join(predictions)
    #print(predction_msg)
    #print(get_keywords(post_content))
    return jsonify({"vertical_type": predction_msg, "keywords": get_keywords(post_content)})
    #return jsonify({"vertical_type": predction_msg})