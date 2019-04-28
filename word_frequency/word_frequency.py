import csv
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
from pandas import DataFrame
import operator

def makeDataframeAndWriteToCsv(freq,vocab,columnName,filename):   
    df=DataFrame(freq,vocab)
    df.columns=[columnName]
    df=df.sort_values(by=columnName,ascending=0)
    df=df.nlargest(500,columnName)

    df.to_csv(filename)
    
words= []
with open('iett6.csv', 'r', encoding="utf8") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
         csv_words = row[1].split(" ")
         for i in csv_words:
              words.append(i)

words=[i.lower() for i in words]
words_counted = []

vectorizer=CountVectorizer()
X=vectorizer.fit_transform(words)
freq=np.ravel(X.sum(axis=0))
vocab=[v[0] for v in sorted(vectorizer.vocabulary_.items(),key=operator.itemgetter(1))]

makeDataframeAndWriteToCsv(freq,vocab,'tf values','tf_list.csv')
