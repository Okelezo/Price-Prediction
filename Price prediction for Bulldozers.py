#!/usr/bin/env python
# coding: utf-8

# # Predicting the sale price of Bulldozers using the Machine Learning  
# This is an end-to-end project using ml.

# 1. Problem Defintion
# 2. Data
# 3. Evaluation
# 4. Getting the tools ready
# 5. Get the ready for modeling
# 6. EDA (explore the data)
# 7. Modeling
# 8. Evaluate your model
# 9. Tune the model
# 10. Predictions using the best model.

# ### Problem Defintion:  
# Our goal is to predict the sale price of bulldozers using the historical data about their features and previous sales.

# ### Data:
# For this competition, you are predicting the sale price of bulldozers sold at auctions.
# 
# The data for this competition is split into three parts:
# 
# 1. Train.csv is the training set, which contains data through the end of 2011.
# 2. Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
# 3. Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.
# 

# ### Evaluation Criteria:  
# The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.

# ### Getting the tools Ready 

# In[10]:


### regular modules for data manipulation
import numpy as np
import pandas as pd

### modules for visualization
import matplotlib.pyplot as plt
import seaborn as sns

## modules for modelling
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV


## magic command for auto-complete
get_ipython().run_line_magic('config', 'completer.use_jedi = False')


# # Getting the data ready:

# ### Data Gathering:

# In[12]:


df = pd.read_csv(r"C:/Users/Okele/Desktop/bluebook/data/Bluebook-for-bulldozers/TrainAndValid.csv", low_memory=False)


# In[13]:


df.tail(10)


# Changing the date format

# In[20]:


df = pd.read_csv(r"C:/Users/Okele/Desktop/bluebook/data/Bluebook-for-bulldozers/TrainAndValid.csv", low_memory=False,parse_dates = ['saledate'])


# In[23]:


df.head()


# In[24]:


##transposed data just to get a different view
df.head().T


# In[30]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


# In[32]:


display_all(df.tail().T)


# In[37]:


display_all(df.describe(datetime_is_numeric=True))


# In[38]:


### information
df.info()


# In[39]:


df.sort_values(by = ['saledate'],inplace = True,ascending = True)
df.saledate


# Adding a few date related columns

# In[27]:


df['SaleYear'] = df.saledate.dt.year
df['SaleMonth'] = df.saledate.dt.month
df['SaleDay'] = df.saledate.dt.day
df['SaleDayOfWeek'] = df.saledate.dt.dayofweek
df['SaleDayOfyear'] = df.saledate.dt.dayofyear


# In[28]:


df.head().T


# In[29]:


df.head()


# #### Issues in the data

# ##### 1. Missing values

# In[40]:


df.isna().any()


# In[43]:


df.isna().sum()


# In[46]:


display_all(df.isnull().sum().sort_index()/len(df))


# ##### 2. Data Types

# In[16]:


df.dtypes


# ### Explore Visualy

# In[41]:


plt.scatter(df.saledate[:1000], df.SalePrice[:1000])


# In[42]:


sns.distplot(df.SalePrice)


# In[13]:


df['saledate'].dtype


# In[14]:


## loading the data again to parse the dates (convert the dates into datetime object instead of string)
df = pd.read_csv('data/TrainAndValid.csv',low_memory=False, parse_dates=['saledate'])


# In[15]:


df.head()


# In[16]:


df.info()


# In[17]:


plt.scatter(df.saledate[:1000], df.SalePrice[:1000])


# In[21]:


df.sort_values(by='saledate', ascending=True, inplace=True)


# In[22]:


df.head()


# In[24]:


df.head(20).T


# In[25]:


df.info()


# ### Dealing with the object types

# #### 1. Deal with the datetime object 

# In[26]:


df.saledate


# In[30]:


df['year'] = df.saledate.dt.year
df['date'] = df.saledate.dt.day
df['month'] = df.saledate.dt.month


# In[31]:


df.info()


# In[33]:


df.drop(['saledate'], axis=1, inplace=True)


# In[35]:


df.info()


# In[38]:


for i in df.columns:
    if df[i].dtype == 'object':
        print(i)


# In[44]:


df.UsageBand.astype('category').cat.as_ordered()


# In[ ]:




