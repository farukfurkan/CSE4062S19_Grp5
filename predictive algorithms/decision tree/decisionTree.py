import numpy as np 
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.cross_validation import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import re 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer

# Function to perform training with entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 3, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
  
  
# Function to make predictions 
def prediction(X_test, clf_object): 
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test)
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      

dataset=pd.read_csv('iett6.csv')

corpus = [] 
  
for i in range(0, len(dataset['Rapor Aciklamasi'])): 
    text = re.sub('[^a-zA-Z]', '', dataset['Rapor Aciklamasi'][i]) 
    text = text.lower() 
    text = text.split() 
    ps = PorterStemmer() 
    text = ''.join(text) 
    corpus.append(text) 

cv = CountVectorizer(max_features = 4000) 
X = cv.fit_transform(corpus).toarray() 
y = dataset.iloc[:, 0].values 

X_train,X_validation,Y_train,Y_validation=train_test_split(X,y,test_size=0.2,random_state=7)


clf_entropy = tarin_using_entropy(X_train,X_validation, Y_train)
    

print("Results Using Entropy:") 
# Prediction using entropy
y_pred_entropy = prediction(X_validation, clf_entropy) 
 
from pandas_ml import ConfusionMatrix
import pandas
pandas.set_option('display.max_colwidth', 15)
pandas.set_option('display.max_columns', 5)
cm = ConfusionMatrix(Y_validation, y_pred_entropy) 
cm.print_stats()
print(cm.stats())
