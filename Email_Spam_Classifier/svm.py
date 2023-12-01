#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm 


# In[24]:


spam = pd.read_csv("spam.csv")


# In[25]:


z = spam["v2"]
y = spam["v1"]
z_train, z_test,y_train, y_test = train_test_split(z,y,test_size = 0.2)


# In[26]:


cv = CountVectorizer()
features = cv.fit_transform(z_train)


# In[27]:


model = svm.SVC()
model.fit(features,y_train)


# In[17]:


features_test = cv.transform(z_test)
print(model.score(features_test,y_test))


# In[ ]:




