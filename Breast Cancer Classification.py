#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd


# In[ ]:





# In[18]:


import numpy as np


# In[ ]:





# In[19]:


import matplotlib.pyplot as plt


# In[ ]:





# In[20]:


import seaborn as sns


# In[ ]:





# In[22]:


from sklearn.datasets import load_breast_cancer


# In[ ]:





# In[23]:


cancer = load_breast_cancer()


# In[ ]:





# In[24]:


cancer


# In[ ]:





# In[25]:


cancer.keys()


# In[ ]:





# In[26]:


print (cancer['DESCR'])


# In[ ]:





# In[27]:


print(cancer['data'])


# In[ ]:





# In[28]:


print(cancer ['target'])


# In[ ]:





# In[29]:


print(cancer['target_names'])


# In[ ]:





# In[30]:


print(cancer['feature_names'])


# In[ ]:





# In[31]:


print(cancer['filename'])


# In[ ]:





# In[32]:


cancer['data'].shape


# In[ ]:





# In[34]:


df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns = np.append(cancer['feature_names'],['target']))


# In[35]:


df_cancer


# In[36]:


df_cancer.head()


# In[38]:


df_cancer.tail()


# In[42]:


sns.pairplot(df_cancer,hue = 'target', vars = ['mean radius','mean texture','mean area','mean perimeter','mean smoothness'])


# In[44]:


sns.countplot(df_cancer['target'])


# In[46]:


sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue='target',data=df_cancer )


# In[48]:


plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(),annot = True)


# In[49]:


x = df_cancer.drop(['target'], axis = 1)


# In[50]:


x


# In[55]:


y= df_cancer['target']


# In[56]:


y


# In[57]:


from sklearn.model_selection import train_test_split


# In[59]:


X_train, X_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 42) 


# In[60]:


X_train


# In[61]:


y_train


# In[62]:


X_test


# In[63]:


from sklearn.svm import SVC


# In[64]:


from sklearn.metrics import classification_report, confusion_matrix


# In[65]:


svc_model = SVC()


# In[68]:


svc_model.fit(X_train,y_train)


# In[70]:


y_predict = svc_model.predict(X_test)


# In[72]:


y_predict


# In[73]:


cm = confusion_matrix(y_test,y_predict)


# In[76]:


sns.heatmap(cm, annot = True)


# In[77]:


min_train = X_train.min()


# In[78]:


range_train = (X_train -min_train).max()


# In[80]:


X_train_scaled = (X_train - min_train)/range_train


# In[81]:


sns.scatterplot(x = X_train['mean area'], y =X_train['mean smoothness'], hue = y_train)


# In[83]:


sns.scatterplot(x = X_train_scaled['mean area'], y =X_train_scaled['mean smoothness'], hue = y_train)


# In[84]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# In[85]:


svc_model.fit(X_train_scaled, y_train)


# In[86]:


y_predict = svc_model.predict(X_test_scaled)


# In[87]:


y_predict


# In[88]:


cm = confusion_matrix(y_test, y_predict)


# In[89]:


cm


# In[90]:


sns.heatmap(cm , annot = True)


# In[92]:


print(classification_report(y_test, y_predict))


# In[120]:


param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001],'kernel' : ['rbf']}


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[121]:


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid,refit = True , verbose = 4)


# In[ ]:





# In[122]:


grid.fit(X_train_scaled, y_train)


# In[123]:


grid.best_params_


# In[124]:


grid_predictions = grid.predict(X_test_scaled)


# In[125]:


cm = confusion_matrix(y_test,grid_predictions )


# In[129]:


sns.heatmap(cm , annot = True)


# In[131]:


print (classification_report(y_test, grid_predictions))






