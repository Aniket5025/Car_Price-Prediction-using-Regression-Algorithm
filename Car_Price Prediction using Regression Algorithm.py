#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# S/O: To predict the cars price is a challanging task because is involve so many factors which can directly affect the price of vehicle,so to automate process buiid a regression model which can predict the vehicle price on the basis of various factors.

# # Libraries

# So we have are importing all important libraries

# In[41]:


#warnings
from warnings import filterwarnings
filterwarnings('ignore')

# data collection
import os
os.chdir("E:\ds\Class_material\Python")
import pandas as pd

#Preprocessing
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

#Model
from sklearn.linear_model import LinearRegression

#Evaluation
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# # Data Collection /reading csv file

# Here our data source is csv file so importing csv file into dataframe format

# In[2]:


df=pd.read_csv('Cars93.csv')
df


# # Define x and y

# Here we are define target variable which are Price

# In[3]:


x=df.drop(['Price','id','Min.Price'],axis=1)
y=df['Price']


# # Exporatory Data Analysis

# In this step we are using some statistics to get the insights of data

# In[4]:


df.head(2)


# In[5]:


df.info()


# In[6]:


df.isna().sum()


# In[7]:


k1=df.corr()['Price']
k1


# In[8]:


k=df.corr()[['Price']]
k


# In[9]:


k.sort_values(by=['Price'],ascending=False)


# In[10]:


# col=k.sort_values(by=['Price'],ascending=False).index
# col


# In[11]:


# cols=col[2:13]
# cols


# In[12]:


# x1=pd.DataFrame(x,columns=cols)
# x1


# # Feature Selection

# In[13]:


x


# In[14]:


for i in x.columns:
    if x[i].dtypes=='object':
        x[i]=x[i].fillna(x[i].mode()[0])
    else:
        x[i]=x[i].fillna(x[i].median())


# In[15]:


cat=x.select_dtypes(include='object')
con=x.select_dtypes(exclude='object')


# In[16]:


cat


# In[17]:


con


# In[18]:


ss=StandardScaler()
le=LabelEncoder()


# In[19]:


con1=pd.DataFrame(ss.fit_transform(con),columns=ss.get_feature_names_out())
con1


# In[20]:


cat1=cat.apply(le.fit_transform)
cat1


# In[21]:


x2=con1.join(cat1)
x2


# In[23]:


lr=LinearRegression()


# In[24]:


sfs=SequentialFeatureSelector(lr,direction='backward')


# In[25]:


sfs.fit_transform(x2,y)


# In[26]:


cols=sfs.get_feature_names_out()
cols


# In[27]:


x3=pd.DataFrame(x,columns=cols)
x3


# In[28]:


x3.info()


# # Pipeline Presprocessing

# In[30]:


cat=[]
con=[]
for i in x3.columns:
    if x3[i].dtypes=='object':
        cat.append(i)
    else:
        con.append(i)


# In[32]:


num_pipe=Pipeline(steps=[('impute',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
cat_pipe=Pipeline(steps=[('imput',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder())])


# In[34]:


pre=ColumnTransformer([('cat_pipe',cat_pipe,cat),('num_pipe',num_pipe,con)])
pre


# In[39]:


x4=pd.DataFrame(pre.fit_transform(x3).toarray(),columns=pre.get_feature_names_out())
x4


# # Split the data

# In[42]:


x_train,x_test,y_train,y_test=train_test_split(x4,y,test_size=0.2,random_state=21)


# # Model fitting

# In[ ]:





# In[43]:


lr.fit(x_train,y_train)


# # Evaluation on Traning data

# In[44]:


y_pred_train=lr.predict(x_train)

mse=mean_squared_error(y_pred_train,y_train)
rmse=mse*0.5
mae=mean_absolute_error(y_pred_train,y_train)
r=r2_score(y_pred_train,y_train)

print('*'*20)
print('MSE:',mse)
print('RMSE:',rmse)
print('MAE:',mae)
print('r2 score:',r)
print('*'*20)


# # Evaluation on Traning data

# In[46]:


y_pred=lr.predict(x_test)

mse1=mean_squared_error(y_pred,y_test)
rmse1=mse1*0.5
mae1=mean_absolute_error(y_pred,y_test)
r1=r2_score(y_pred,y_test)

print('*'*20)
print('MSE:',mse1)
print('RMSE:',rmse1)
print('MAE:',mae1)
print('r2 score:',r1)
print('*'*20)


# # Unseen data Evaluation

# In[47]:


df2=pd.read_csv('sample.csv')
df2


# In[50]:


x5=pd.DataFrame(pre.transform(df2).toarray(),columns=pre.get_feature_names_out())
x5


# In[52]:


y_pred1=lr.predict(x5)
y_pred1


# In[54]:


predict=df2[['Model','Price']]
predict


# In[55]:


predict['Prediction']=y_pred1
predict


# In[56]:


predict.to_csv('Prediction.csv',index=False)


# In[ ]:




