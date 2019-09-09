
from flask import jsonify,request
from pprint import pprint
import simplejson as json
import sys 
from nltk.corpus import brown
from nltk.corpus import reuters
import nltk
from nltk.corpus import PlaintextCorpusReader

import json
from difflib import get_close_matches


def get_trigram_freq(tokens):
    tgs = list(nltk.trigrams(tokens))

    a,b,c = list(zip(*tgs))
    bgs = list(zip(a,b))
    return nltk.ConditionalFreqDist(list(zip(bgs, c)))

def get_bigram_freq(tokens):
    bgs = list(nltk.bigrams(tokens))

    return nltk.ConditionalFreqDist(bgs)

def appendwithcheck (preds, to_append):
    for pred in preds:
        if pred[0] == to_append[0]:
            return
    preds.append(to_append)

def incomplete_pred(words, n):
    all_succeeding = bgs_freq[(words[n-2])].most_common()
    #print (all_succeeding, file=sys.stderr)
    preds = []
    number=0
    for pred in all_succeeding:
        if pred[0].startswith(words[n-1]):
            appendwithcheck(preds, pred)
            number+=1
        if number==3:
            return preds
    if len(preds)<3:
        med=[]
        for pred in all_succeeding:
            med.append((pred[0], nltk.edit_distance(pred[0],words[n-1], transpositions=True)))
        med.sort(key=lambda x:x[1])
        index=0
        while len(preds)<3:
            #print (index, len(med))
            if index<len(med):
                if med[index][1]>0:
                    appendwithcheck(preds, med[index])
                index+=1
            if index>=len(preds):
                return preds

    return preds

new_corpus = PlaintextCorpusReader('./','.*')

#tokens = nltk.word_tokenize(raw)
tokens = brown.words() + new_corpus.words('data/large_blob-cms.txt')
#tokens = reuters.words()

#compute frequency distribution for all the bigrams and trigrams in the text
bgs_freq = get_bigram_freq(tokens)
tgs_freq = get_trigram_freq(tokens)

def translate(word):
    data = json.load(open("data/data.json"))
    word = word.lower()
    if word in data:
        for d in data:
            #return data[word]
            return False
    elif word.title() in data:
        #return data[word.title()]
        return False
    elif word.upper() in data:
        #return data[word.upper()]
        return False
    else:
        return get_close_matches(word, data.keys())[0]


def predict_word(work, string):
    words=string.split()
    #print(words)
    n = len(words)
    
    output_data = []

    if work=='pred':
        if n == 1:
            print('**** 1 ****')
            #print(bgs_freq[(string)].most_common(5))
            print(bgs_freq[(string)].most_common(5))
            return json.dumps(bgs_freq[(string)].most_common(5))
          
        elif n>1:
            print('**** 2 ****')
            #print(tgs_freq[(words[n - 2], words[n - 1])].most_common(5))
            print(tgs_freq[(words[n-2],words[n-1])].most_common(5))
            return json.dumps(tgs_freq[(words[n-2],words[n-1])].most_common(5))
    else:
        print('**** 3 ****')
        #print(tgs_freq[(words[n - 2], words[n - 1])].most_common(5))
        print(incomplete_pred(words, n))
        return jsonify({"predictions": incomplete_pred(words, n), "spell_suggestion": translate(words[0])})
        #return json.dumps(incomplete_pred(words, n))