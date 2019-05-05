import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="ticks")


####Plotting class histogram
data = {'Müsbet': {1:48702}, 'Menfi': {1:3189}, 'Uyarı': {1:14567 }}

df = pd.DataFrame(data)

df.plot(kind='bar')

plt.xlabel('Classes')
plt.ylabel('Number of complaint')
plt.show()


####Plotting word-frequency histogram
data = pd.read_csv("tf_list2.csv")  
data.head()

df = pd.DataFrame(data)
df.plot(kind='bar')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()


####Plotting Scatter plot matrix
data = pd.read_csv("tf_list.csv")  
data.head()

df = pd.DataFrame(data)


firstGroup=[]
secondGroup=[]
thirdGroup=[]
fourthGroup=[]
fifthGroup=[]
counter=0
for i in df['tf values']:
    if(counter<100):
        firstGroup.append(df['tf values'][counter])
    elif(99<counter<200):
        secondGroup.append(df['tf values'][counter])
    elif(199<counter<300):
        thirdGroup.append(df['tf values'][counter])
    elif(299<counter<400):
        fourthGroup.append(df['tf values'][counter])
    elif(399<counter<500):
        fifthGroup.append(df['tf values'][counter])
    counter+=1

df2 = pd.DataFrame()

df2['1. Group']=firstGroup
df2['2. Group']=secondGroup
df2['3. Group']=thirdGroup
df2['4. Group']=fourthGroup
df2['5. Group']=fifthGroup

pd.plotting.scatter_matrix(df2)
plt.show()
