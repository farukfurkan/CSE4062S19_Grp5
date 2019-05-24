# cleaning texts 
import pandas as pd 
import re 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import CountVectorizer 
  

dataset=pd.read_csv('iett6.csv')
           
corpus = [] 
  
for i in range(0, len(dataset['Rapor Aciklamasi'])): 
    text = re.sub('[^a-zA-Z]', '', dataset['Rapor Aciklamasi'][i]) 
    text = text.lower() 
    text = text.split() 
    ps = PorterStemmer() 
    text = ''.join(text) 
    corpus.append(text) 

# creating bag of words model 
cv = CountVectorizer(max_features = 4000) 
X = cv.fit_transform(corpus).toarray() 
y = dataset.iloc[:, 0].values 
##print(y)

# splitting the data set into training set and test set 
from sklearn.cross_validation import train_test_split 
  
X_train, X_test, y_train, y_test = train_test_split( 
           X, y, test_size = 0.2, random_state = 7) 


# fitting naive bayes to the training set 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import confusion_matrix


classifier = GaussianNB(); 
classifier.fit(X_train, y_train)


  
# predicting test set results 
y_pred = classifier.predict(X_test) 
  
# making the confusion matrix 
##cm = confusion_matrix(y_test, y_pred) 
##print(cm)
##
##c=0
##a=0
##for i in y_pred:
####    print(i,y_test[c])
##    if(i==y_test[c]):
##        a+=1
##    c+=1
##print(a/c)

from pandas_ml import ConfusionMatrix
import pandas
pandas.set_option('display.max_colwidth', 15)
pandas.set_option('display.max_columns', 5)
cm = ConfusionMatrix(y_test, y_pred) 
cm.print_stats()
print(cm.stats())
