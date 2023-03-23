#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#dataframe
df=pd.read_csv('Advertising.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.info


# In[6]:


df.describe


# In[7]:


df.columns


# In[8]:


#checking duplicates
df.duplicated().sum()


# In[10]:


plt.figure(figsize=(5,5))
sns.scatterplot(data=df,x=df['TV'],y=df['Sales'])
plt.show()


# In[11]:


plt.figure(figsize=(4,4))
sns.scatterplot(data=df,x=df['Radio'],y=df['Sales'])
plt.show()


# In[12]:


plt.figure(figsize=(4,4))
sns.scatterplot(data=df,x=df['Newspaper'],y=df['Sales'])
plt.show()


# In[13]:


X=df.drop('Sales',axis=1)


# In[14]:


X


# In[15]:


y=df['Sales']


# In[16]:


y


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.20, random_state = 0)


# In[19]:


from sklearn.linear_model import LinearRegression
model= LinearRegression()


# In[20]:


#fitting the model to the dataset
model.fit(X_train,y_train)


# In[21]:


#predictions
y_predictions=model.predict(X_test)


# In[22]:


y_predictions


# In[23]:


# Lets evaluate the model for its accuracy using various metrics such as RMSE and R-Squared
from sklearn import metrics

print('MAE:',metrics.mean_absolute_error(y_predictions,y_test))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_predictions,y_test)))
print('R-Squared',metrics.r2_score(y_predictions,y_test))


# In[ ]:




