import csv
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np
from pandas import DataFrame
import pandas as pd
from TurkishStemmer import TurkishStemmer
import trstop
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', 15)
pd.set_option('display.max_columns', 5)

df=pd.read_csv('iett6.csv')


df['Rapor Aciklamasi']=df['Rapor Aciklamasi'].str.lower()

vectorizer = TfidfVectorizer(stop_words=set(stopwords.words('turkish')))
X=vectorizer.fit_transform(df['Rapor Aciklamasi']).todense()

#### finding best k value for  k mean algorithm
pca=PCA(n_components=2).fit(X)
data2D=pca.transform(X)

distortions=[]
K=range(1,len(df['Rapor Aciklamasi']))
true_k=len(df['Rapor Aciklamasi'])
threshold=0.15
k_list=[]
for k in K:
  kmeanModel=KMeans(n_clusters=k).fit(data2D)
  kmeanModel.fit(data2D)
  distortions.append(sum(np.min(cdist(data2D,kmeanModel.cluster_centers_,'euclidean'),axis=1))/data2D.shape[0])
  if(distortions[k-1]<=threshold):
    true_k=k
    threshold=0.000000000001
  k_list.append(k)
  print(k,distortions[k-1])
  if(k==10):
      break

##print(true_k)
plt.plot(k_list,distortions,'-bx')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

##Kmean model
model=KMeans(n_clusters=true_k,init='k-means++',max_iter=100,n_init=1)
model.fit(X)

log={'cluster 1':0,'cluster 2':0,'cluster 3':0}
for d in df['Rapor Aciklamasi']:
  Y=vectorizer.transform([d])
  prediction=model.predict(Y)
  if(prediction[0]==0):log['cluster 1']+=1
  elif(prediction[0]==1):log['cluster 2']+=1
  elif(prediction[0]==2):log['cluster 3']+=1

print(log)
##data = np.array([log[i] for i in log])
##print (data.std())
##avg=np.average(data)
##s=np.array([(x-avg)**2 for x in data])
##print(s)
##s=np.sum(s)
##print(s)
