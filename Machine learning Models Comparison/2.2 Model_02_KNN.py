#!/usr/bin/env python
# coding: utf-8

# ## Importing Necessary Packages

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ## Importing Dataset

# In[2]:


df_ori = pd.read_excel("heart_disease.xlsx")
df = df_ori


# In[3]:


#feature description
df_desc = pd.read_excel("heart_disease.xlsx","Details")
df_desc


# Dataset Imported Before Data Exploration

# In[4]:


df_ori


# ## Data Exploration

# In[5]:


obs, feat = df.shape
print("No. of observations: {}".format(obs))
print("No. of features: {}".format(feat))


# In[6]:


df.info()


# Separate the dataframe to features X and target y

# In[7]:


X = df.drop(["age","sex","cp","trestbps","chol","fbs","restecg",
             "thalach","exang","oldpeak","slope","ca","thal"], axis = 1)
Y = df["target"]


# ## Normalize Data

# Data Standardization gives the data zero mean and unit variance, it is a good practice, especially for algorithms such as KNN which is based on the distance of data points:

# To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:

# In[8]:


X = df[["age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal"]].values


# In[9]:


from sklearn.preprocessing import StandardScaler

# using standardscaler to scale the data
scaler = StandardScaler().fit(X)
X = scaler.transform(X.astype(float))


# ## Train Test Split

# Split the data into train and test set of 80:20.

# In[10]:


from sklearn.model_selection import train_test_split

#split the data into 80:20
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    test_size = 0.20, 
                                                    random_state = 0)

print ('Train set:', X_train.shape,  Y_train.shape)

print ('Test set:', X_test.shape,  Y_test.shape)


# ## Building KNN Model

# Import library

# In[11]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


# Investigating the best k for this 80:20 Train & Test data split by plotting k-NN classification accuracy across k values from 1 to 20.

# In[12]:


Ks = 21
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,Y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = accuracy_score(Y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==Y_test)/np.sqrt(yhat.shape[0])

print ("Listed below are the accuracy of the model respective to the value of K:")
mean_acc


# Plotting the values into line chart.

# In[13]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.legend('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


# To support the result of the best k from the line chart, we investigate further by comparing the mean error values resulted from each K.

# In[14]:


error = []
# Calculating error for K values between 1 and 20
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    
    #finding the mean of the scenario when results of test 
    #not identical to value predicted by the model
    
    error.append(np.mean(pred_i != Y_test))
    
plt.figure(figsize=(12, 6))
plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
print("Minimum error:-",min(error),"at K =",error.index(min(error))+1)


# Based on the investigations above, we found that :

# In[15]:


print("Minimum error:-",min(error),"at k =",error.index(min(error))+1)
print( "The best accuracy obtained", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# ## Checking if this is a good model

# Based on the result on accuracy for K values of 1 to 20, it seems that K=9 achieved the most accuracy.<br>
# So, in this section, we are checking further into the model to compare the train and test results into :
# <br>
# 1. Precision
# <br>
# 2. Recall
# <br>
# 3. F1
# <br>
# 4. KNN Accuracy
# <br>
# 5. Confusion Matrix
# <br>

# Evaluating model.

# In[16]:


colors = ['#F93822','#FDD20E']


def model_evaluation(classifier):
    
    
    knn = KNeighborsClassifier(n_neighbors=classifier)    
    y_proba_knn = knn.fit(X_train, Y_train).predict_proba(X_test)
    knn_predict = knn.predict(X_test)
    #roc = roc_auc_score(Y_test, y_proba_knn[:,1])
    
    # Confusion Matrix
    cm = confusion_matrix(Y_test,knn_predict)
    
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
    print(classification_report(Y_test,knn_predict))
    print('KNN Accuracy= {:.2f}'.format(accuracy_score(Y_test, knn_predict)))


# In[17]:


selected_k = mean_acc.argmax()+1
print ("Evaluate the model for K = ", selected_k)
model_evaluation(selected_k)


# In[ ]:




