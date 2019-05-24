import pandas
import sys
import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import os
import datetime
from sklearn import model_selection

non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
pandas.set_option('display.max_colwidth', 15)
pandas.set_option('display.max_columns', 5)

data = pandas.read_csv('iett6.csv', sep=",", header=None)
data.columns = ["Durum", "Rapor Aciklamasi"]


#print(data)
X_train,X_validation,Y_train,Y_validation=model_selection.train_test_split(data['Rapor Aciklamasi'].values,data.iloc[:, 0].values,test_size=0.2,random_state=7)
print(len(X_train),len(Y_train))
stemmer=LancasterStemmer()
##training_data=[]
##for i in range(0,len(X_train)):
##  training_data.append({"class":Y_train[i],"sentence":X_train[i]})
##
##print("%s sentences in training data" %len(training_data))
##
##words=[]
##classes=[]
##documents=[]
##ignore_words=['?']
##
##for pattern in training_data:
##  w=nltk.word_tokenize(pattern['sentence'])
##  words.extend(w)
##  documents.append((w,pattern['class']))
##  if pattern['class'] not in classes:
##    classes.append(pattern['class'])
##
##words=[stemmer.stem(w.lower()) for w in words if w not in ignore_words]
##words=list(set(words))
##
##classes=list(set(classes))
##
##print(len(documents),'documents')
##print(len(classes),'classes',classes)
##print(len(words),'unique stemmed words',words)
##
##training=[]
##output=[]
##
##output_empty=[0]*len(classes)
##
##for doc in documents:
##  bag=[]
##  pattern_words=doc[0]
##  pattern_words=[stemmer.stem(word.lower()) for word in pattern_words]
##
##  for w in words:
##    bag.append(1) if w in pattern_words else bag.append(0)
##
##  training.append(bag)
##
##  output_row=list(output_empty)
##  output_row[classes.index(doc[1])]=1
##  output.append(output_row)
##
##i=0
##w=documents[i][0]
##print([stemmer.stem(word.lower()) for word in w])
##print(training[i])
##print(output[i])

import numpy as np
import time

def sigmoid(x):
##  x=np.clip(x,-500,500)
  output = 1/(1+np.exp(-x))
  return output

def sigmoid_output_to_derivative(output):
  return output*(1-output)

def clean_up_sentence(sentence):
  sentence_words=nltk.word_tokenize(sentence)
  sentence_words=[stemmer.stem(word.lower()) for word in sentence_words]
  return sentence_words

def bow(sentence,words,show_details=False):
  sentence_words=clean_up_sentence(sentence)
  bag=[0]*len(words)

  for s in sentence_words:
    for i,w in enumerate(words):
      if w == s:
        bag[i] = 1
        if show_details:
          print("found in bag: %s" %w)

  return(np.array(bag))

def think(sentence,show_details=False):
  x=bow(sentence.lower(),words,show_details)
  if show_details:
    print("sentence:",sentence,"\n bow",x)

  l0=x
  l1=sigmoid(np.dot(l0,synapse_0))
  l2=sigmoid(np.dot(l1,synapse_1))

  return l2

def train(X,y,hidden_neurons=10,alpha=1,epochs=50000,dropout=False,dropout_percent=0.5):
  print("Training with %s neurons, alpha:%s, dropuot:%s %s"%(len(X),str(alpha),dropout,dropout_percent if dropout else ''))
  print("Input matrix:%sx%s    Output matrix:%sx%s" %(len(X),len(X[0]),1,len(classes)))
  np.random.seed(1)

  last_mean_error = 1

  synapse_0 = 2*np.random.random((len(X[0]),hidden_neurons))-1
  synapse_1 = 2*np.random.random((hidden_neurons,len(classes)))-1

  prev_synapse_0_weight_update = np.zeros_like(synapse_0)
  prev_synapse_1_weight_update = np.zeros_like(synapse_1)

  synapse_0_direction_count = np.zeros_like(synapse_0)
  synapse_1_direction_count = np.zeros_like(synapse_1)

  for j in iter(range(epochs+1)):
    print(j)
    layer_0=X
    layer_1=sigmoid(np.dot(layer_0,synapse_0))

    if(dropout):
      layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0]

    layer_2 = sigmoid(np.dot(layer_1,synapse_1))

    layer_2_error = y - layer_2

    if(j%10000)==0 and j > 5000:
      if np.mean(np.abs(layer_2_error)) < last_mean_error:
        print("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))))
        last_mean_error = np.mean(np.abs(layer_2_error))
      else:
        print("break:",np.mean(np.abs(layer_2_error)),">",last_mean_error)
        break
    
    layer_2_delta = layer_2_error*sigmoid_output_to_derivative(layer_2)

    layer_1_error = layer_2_delta.dot(synapse_1.T)
    layer_1_delta = layer_1_error*sigmoid_output_to_derivative(layer_1)

    synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
    synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

    if(j>0):
      synapse_0_direction_count += np.abs(((synapse_0_weight_update >0)+0)-((prev_synapse_0_weight_update >0)+0))
      synapse_1_direction_count += np.abs(((synapse_1_weight_update >0)+0)-((prev_synapse_1_weight_update >0)+0))

    synapse_1 += alpha * synapse_1_weight_update
    synapse_0 += alpha * synapse_0_weight_update

    prev_synapse_0_weight_update = synapse_0_weight_update
    prev_synapse_1_weight_update = synapse_1_weight_update

  now=datetime.datetime.now()

  synapse = {'synapse0':synapse_0.tolist(),'synapse1':synapse_1.tolist(),
             'datetime':now.strftime("%Y-%m-%d %H:%M"),
             'words':words,
             'classes':classes}

  synapse_file = "synapse.json"

  with open(synapse_file,'w') as outfile:
    json.dump(synapse,outfile,indent=4,sort_keys=True)
  print("saved synapses to:",synapse_file)

##X=np.array(training)
##y=np.array(output)
##
##start_time=time.time()
##
##train(X,y,hidden_neurons=50,alpha=0.001,epochs=100000,dropout=False,dropout_percent=0.2)
##
##elapsed_time=time.time()- start_time
##print("processing time:",elapsed_time,"seconds")

ERROR_THRESHOLD = 0.1
synapse_file = 'synapse.json'
with open(synapse_file) as data_file:
  synapse = json.load(data_file)
  synapse_0 = np.asarray(synapse['synapse0'])
  synapse_1 = np.asarray(synapse['synapse1'])
  words=np.asarray(synapse['words'])
  classes=np.asarray(synapse['classes'])

def classify(sentence, show_details=False):
  results = think(sentence,show_details=False)

  results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
  results.sort(key=lambda x: x[1], reverse=True)
  return_results=[[classes[r[0]],r[1]] for r in results]
##  print ("%s \n classification: %s" % (sentence, return_results))
  return return_results

##classify("we have to run to escape from him")
##word=""
##while(word!='exit'):
##  word=input("test=>")
##  classify(word)

from sklearn.metrics import classification_report
Y_predict=[]
c=0
for i in range(0,len(X_validation)):
  Y_predict.append(classify(X_validation[i]))
  if(len(Y_predict[i])!=0):
##    print(Y_validation[i],Y_predict[i][0])
    if(len(Y_predict[i])==2):
      if(Y_predict[i][0][0]==Y_validation[i] or Y_predict[i][1][0]==Y_validation[i]):
##        print("yyyyyyyyyyyy")
        c=c+1
    else:
      if(Y_predict[i][0][0]==Y_validation[i]):
##        print("zzzzzzzzzzzz")
        c=c+1

print('accuracy=',c/len(X_validation))

##from sklearn.metrics import f1_score
##f1_score(Y_validation[i], Y_predict[i][0][0], average=None)
##print(f1_score)

