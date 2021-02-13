#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection

# Importing Dataset

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


# In[3]:


creditcarddata = pd.read_csv('C:/Users/mohit/OneDrive/Desktop/Northeastern/ALY 6110 Big data/Week 3/raghu543-credit-card-fraud-data/data/creditcard.csv')


# In[4]:


creditcarddata.head()


# In[6]:


creditcarddata.tail()


# In[7]:


creditcarddata.describe()


# In[8]:


pd.set_option('precision',3)
creditcarddata.loc[:, ['time', 'amount']].describe()


# In[9]:


#visualizations of time and amount
plt.figure(figsize=(10,8))
plt.title('Distribution of Time Feature')
sns.distplot(creditcarddata.time)


# In[10]:


creditcarddata.info()


# In[11]:


creditcarddata.isnull().sum()


# In[12]:


creditcarddata.sample(5)


# In[13]:


plt.figure(figsize=(10,8))
plt.title('Distribution of Monetary value feature')
sns.distplot(creditcarddata.amount)


# Mean in the dataset =88 dollars
# Biggest transaction a monetary value of around 25,691 dollars

# # Fraud vs Normal Transactions 

# In[14]:


counts = creditcarddata['class'].value_counts()
normal = counts[0]
fraudulent = counts[1]
form_norm = (normal/(normal+fraudulent))*100
form_fraud = (fraudulent/(normal+fraudulent))*100
print('There were {} non-fraudulent transactions ({:.3f}%) and {} fraudulent transactions ({:.3f}%).'.format(normal, form_norm, fraudulent, form_fraud))


# In[15]:


cre = creditcarddata.corr()
cre


# In[16]:


#heatmap
cre = creditcarddata.corr()
plt.figure(figsize=(12, 10))
mmap = sns.heatmap(data = cre, cmap = "Greens")
plt.title("Correlation Heatmap")


# # Highest correlation comes from:
# * Time & V3(-0.42)
# * Amount & V2(-0.53)
# * Amount & V4(0.4)
# 
# 
# While these correlations are high, I don't expect it to run the risk of multicollinearity.

# In[17]:


creditcarddata.replace(False, 0, inplace = True)
creditcarddata.replace(True, 1, inplace = True)


# In[18]:


plt.figure(figsize=(10,6))
creditcarddata[creditcarddata['class']==1]['amount'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Non-Fraudulent')
creditcarddata[creditcarddata['class']==0]['amount'].hist(alpha=0.5,color='red',
                                              bins=30,label='Fraud')
plt.legend()
plt.xlabel('AMOUNT')


# In[19]:


sns.countplot(x='class',data=creditcarddata)


# # Skewness

# In[20]:


skew_ = creditcarddata.skew()
skew_


# # Scaling Amount and Time
# 
# * StandardScaler: It transforms the data in such a manner that it has mean as 0 and standard deviation as 1. In short, it standardizes the data. Standardization is useful for data which has negative values. It arranges the data in normal distribution. It is more useful in classification than regression.
# 
# * Standardscaler standardizes a feature by subtracting the mean and then scaling to unit variance. Unit variance means divide all the values by the standard deviation.
# 
# * Normalizer: It squeezes the data between 0 and 1. It performs normalization. Due to the decreased range and magnitude, the gradients in the training process do not explode and you do not get higher values of loss. Is more useful in regression than classification.

# In[21]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler2 = StandardScaler()
#Scaling Time
scaled_time = scaler.fit_transform(creditcarddata[['time']])
flat_list1 = [item for sublist in scaled_time.tolist() for item in sublist]
scaled_time = pd.Series(flat_list1)


# In[22]:


#Scaling Amount
scaled_amount = scaler2.fit_transform(creditcarddata[['amount']])
flat_list2 = [item for sublist in scaled_amount.tolist() for item in sublist]
scaled_amount = pd.Series(flat_list2)


# Concatenating newly converted columns with original data i.e.creditcarddata

# In[23]:


creditcarddata = pd.concat([creditcarddata, scaled_amount.rename('scaled_amount'), scaled_time.rename('scaled_time')], axis=1)
creditcarddata.sample(5)


# In[24]:


#dropping old amount and time columns
creditcarddata.drop(['amount', 'time'], axis=1, inplace=True)


# # Train Test Split
# Now its time to split our data into a training set and a testing set!
# 
# ** Use sklearn to split your data into a training set and a testing set!

# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


X = creditcarddata.drop('class', axis=1)
y = creditcarddata['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# # Training a decision Tree
# Decision tree and their ensemble are popular methods for machine learning tasks of classification and regression. Decision trees are widely used since they are easy to interpret, handle, categorical features, extend to the multiclass classification setting, do not require feature scaling, and are able to capture non-linearities and feature interactions. Tree ensemble algorithms such as random forests and boosting are among the top performers for classification and regression tasks

# In[27]:


from sklearn.tree import DecisionTreeClassifier


# In[28]:


dtree = DecisionTreeClassifier()


# In[29]:


dtree.fit(X_train,y_train)


# # Predictions and Evaluation of Decision Tree
# **Create predictions from the test set and create a classification report and a confusion matrix.**

# In[30]:


predictions = dtree.predict(X_test)


# In[31]:


from sklearn.metrics import classification_report,confusion_matrix


# In[32]:


print(classification_report(y_test,predictions))


# In[33]:


print(confusion_matrix(y_test,predictions))


# # Training the Random Forest Model
# Now its time to train RF!

# In[34]:


from sklearn.ensemble import RandomForestClassifier


# In[35]:


rfc = RandomForestClassifier(n_estimators=600)


# In[36]:


rfc.fit(X_train, y_train)


# # Prediction and Evaluations

# In[37]:


predictions = rfc.predict(X_test)


# **Now create a classification report from the results**

# In[38]:


from sklearn.metrics import classification_report,confusion_matrix


# In[39]:


print(classification_report(y_test,predictions))


# In[40]:


print(confusion_matrix(y_test,predictions))


# **Dimensionality Reduction - 
# It will less mislead the data means model accuracy increases, 
# Less Dimensions means less computing, 
# less data means that algorithms train faster, 
# less data means less storage space required, 
# less dimensions allow usage of algorithms unfit for a large number for dimensions, 
# Removes redundant features and noise.**

# In[ ]:


# from sklearn.manifold import TSNE

# X = creditcarddata.drop('class', axis=1)
# y = creditcarddata['class']


# In[ ]:


# #t-sne
# X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)


# # Classification Algorithms

# In[60]:


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[61]:


#train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[62]:


X_train = X_train.values
X_validation = X_test.values
y_train = y_train.values
y_validation = y_test.values


# In[63]:


print('X_shapes:\n', 'X_train:', 'X_validation:\n', X_train.shape, X_validation.shape, '\n')
print('Y_shapes:\n', 'Y_train:', 'Y_validation:\n', y_train.shape, y_validation.shape)
