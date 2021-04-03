#!/usr/bin/env python
# coding: utf-8

# # IMPORT NECESSARY MODELS

# In[16]:


import tensorflow
### Data Collection
import pandas_datareader as pdr
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# In[17]:


### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.models import load_model
import keras


# In[3]:


import joblib,json


# In[4]:


#from sklearn.externals import joblib


# In[4]:


key="USE TIINGO TO GET THE API KEY AND PASTE IT HERE"


# In[5]:





# In[6]:


# READ DATA AND SAVE IT TO CSV


# In[7]:


df = pdr.get_data_tiingo('AAPL', api_key=key)


# In[8]:


df.to_csv('AAPL.csv')


# In[9]:


# READ DATA


# In[8]:


df=pd.read_csv('AAPL.csv')


# In[48]:


df1=df.reset_index()['close']


# In[49]:


### LSTM are sensitive to the scale of the data. so we apply MinMax scaler


# In[50]:


scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[51]:


##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[52]:


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[54]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[55]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[25]:


# the model


# In[26]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[27]:


hist = model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[56]:


#model.save('my_model2.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model
# returns a compiled model
# identical to the previous one
model1 = load_model('my_model2.h5')


# In[57]:


### Lets Do the prediction and check performance metrics
train_predict=model1.predict(X_train)
test_predict=model1.predict(X_test)


# In[58]:


train_predict[:5]


# In[59]:


test_predict[:5]


# In[60]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[61]:


train_predict[:5]


# In[62]:


test_predict[:5]


# In[ ]:




