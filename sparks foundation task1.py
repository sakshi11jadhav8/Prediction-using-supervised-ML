#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation - GRIP - Data Science and Business Analytics - NOV'2022

# # TASK 1 : Prediction using supervised ML

# ### Author : Sakshi Jadhav
# ### Dataset used : Student Scores
# * It can be downloaded through the following link - http://bit.ly/w-data
# 

# ### Problem Statement(s):
# * Predict the percentage of a student based on the no. of students
# * What will be predicted score if a studies for 9.25 hrs/day?

# ### Import necessary libraries

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn


# ### Read the csv dataset as a pandas dataframe

# In[5]:


data = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')


# In[6]:


data


# In[8]:


data.head()


# In[9]:


data.shape


# In[10]:


data.info()


# In[11]:


data.describe()


# In[12]:


data.isnull().sum()


# ### Visualize the data

# In[13]:


## Checking if there is any ouliers in the data set or not

data['Hours'].plot.box()


# In[14]:


data['Scores'].plot.box()


# In[16]:


## plotting the data into scatter plot to see the relationship

data.plot(kind = 'scatter', x = 'Hours', y = 'Scores')
plt.show()


# In[17]:


## Above plot shows that fairly relatioship is there
## Let's check the relatioship with correlation matrix

data.corr(method = 'pearson')


# In[18]:


data.corr(method = 'spearman')


# ## Linear Regression

# In[19]:


x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


# In[21]:


## Splitting the dataset into Testing and Training

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)


# In[22]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)


# In[23]:


m = reg.coef_
c = reg.intercept_
line = m*x+c
plt.scatter(x, y)
plt.plot(x, line)
plt.show()


# ### Visualisng the model

# In[24]:


plt.rcParams['figure.figsize']=[8,6]
plt.scatter(x_train,y_train,color='red')
plt.plot(x,line,color = 'green')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[28]:


print(x_test)
y_pred = reg.predict(x_test)


# In[29]:


y_test


# In[30]:


y_pred


# In[31]:


data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
data


# ### What will be the predicted score if a student studies 9.25 Hours/Day ? 

# In[32]:


new_hours = 9.25
new_pred = reg.predict([[new_hours]])
print('Predicted Score = {}'.format(new_pred[0]))


# ## Model Evaluation

# In[34]:


from sklearn import metrics 
from sklearn.metrics import r2_score
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('R2 Score', r2_score(y_test, y_pred))


# In[ ]:




