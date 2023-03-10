#!/usr/bin/env python
# coding: utf-8

# In[1]:


#package for read data visualization
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#read data file
df = pd.read_csv("heart_disease.csv")


# In[3]:


#import the decision tree package, Machine learning Algorithm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[4]:


#checking the data is correct or not
print(df.shape)


# In[5]:


#print the data from csv file
df.head()


# In[6]:


#rename the data of the column from shortform name
df.columns=['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achived', 
            'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemla', 'target']


# In[7]:


#checking the data is completed changed or not
df.info()


# In[8]:


#redefining categorical variables for better understanding
#that is not error is just warning that value is trying to be set on a copy of a slice from a DataFrame

df['sex'][df['sex'] == 0] = 'female'
df['sex'][df['sex'] == 1] = 'male'

df['chest_pain_type'][df['chest_pain_type'] == 0] = 'typical angina'
df['chest_pain_type'][df['chest_pain_type'] == 1] = 'atypical angina'
df['chest_pain_type'][df['chest_pain_type'] == 2] = 'non-angina pain'
df['chest_pain_type'][df['chest_pain_type'] == 3] = 'asymptomatic'

df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

df['exercise_induced_angina'][df['exercise_induced_angina'] == 0] = 'no'
df['exercise_induced_angina'][df['exercise_induced_angina'] == 1] = 'yes'

df['st_slope'][df['st_slope'] == 1] = 'upsloping'
df['st_slope'][df['st_slope'] == 2] = 'flat'
df['st_slope'][df['st_slope'] == 3] = 'downsloping'

df['thalassemla'][df['thalassemla'] == 1] = 'normal'
df['thalassemla'][df['thalassemla'] == 2] = 'fixed defect'
df['thalassemla'][df['thalassemla'] == 3] = 'reversable defect'


# In[9]:


#check again the data is in object
df.dtypes


# In[10]:


#categerical variables
#to convert data in right format
obj_cats = ['sex','chest_pain_type','fasting_blood_sugar','exercise_induced_angina','st_slope', 'num_major_vessels', 'thalassemla']

for colname in obj_cats:
    df[colname] = df[colname].astype('category')


# In[11]:


#checking the object is in category
df.info()


# In[12]:


#creating dummies for the data
df = pd.get_dummies(df, drop_first=True)


# In[13]:


#checking data and read data
df.head()


# In[14]:


#visualisation the data
sns.countplot(x='target', data=df, palette='bwr')
plt.show()


# In[15]:


#then defining attributes and target variable of the data
X = df.drop(['target'], axis=1)
y = df['target']


# In[16]:


#the we need splitting the data for doing modeling into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[17]:


colors = ['#F93822','#FDD20E']


 
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
#roc = roc_auc_score(Y_test, y_proba_knn[:,1])
    
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
    
# Labelling the matrix
names = ['True Neg','False Pos','False Neg','True Pos']
    
# Counts of the test data and labelling it on the matrix
counts = [value for value in cm.flatten()]
    
# Get the % of the grand total on the matrix
percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm,annot = labels,cmap = colors,fmt ='')
    
# Classification Report
print(classification_report(y_test, y_pred))

