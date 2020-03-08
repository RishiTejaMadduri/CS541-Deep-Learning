#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np


# In[15]:


x_tr=appending("age_regression_Xtr (1).npy")
y_tr=np.load("age_regression_ytr.npy")
x_te=appending("age_regression_Xte.npy")
y_te=np.load("age_regression_yte.npy")

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x_tr, y_tr, test_size = 0.20, random_state = 0)

#Mapping the x values to y values
mapp = list(zip(X_tr, ytr))
np.random.shuffle(mapp)
#Unzipping the values
x_tr,ytr=zip(*mapp)


# In[13]:


def appending(X):
    x_tr=np.reshape(np.load(X),(-1,48*48))
    return np.c_[x_tr,np.ones(x_tr.shape[0])]


# In[62]:


lr=[0.01,0.0001,0.00001,0.000001]
alpha=[0.1,0.01,0.001,0.0005]
batchsize=[10,50,200,100]
epoch=[50,100,500,750]


# In[63]:


def G_D(lr,alpha,batchsize,epoch,X,Y):
    w=np.random.randn(x_train.shape[1])
    mini_x=[]
    mini_y=[]
    for i in range(0,len(x_train),batchsize):
        if i ==0:
            continue        
        else:            
                mini_x.append(x_train[i-batchsize:i])
                mini_y.append(y_train[i-batchsize:i])
    mini_x.append(x_train[len(x_train)-batchsize:len(x_train)])
    mini_y.append(y_train[len(y_train)-batchsize:len(y_train)])
    for i in range(epoch):     
        for j in range(len(mini_x)):          
            GD=np.dot(mini_x[j].T,(np.dot(mini_x[j],w)-mini_y[j]))/(mini_x[j].shape[0]) + (alpha/mini_x[j].shape[0])*w
            w-=lr*GD
    return w


# In[65]:


Hyperparams={}
lr_len=len(lr)
for k in range(lr_len):
    for l in range(lr_len):
        for m in range(lr_len):
              for n in range(lr_len):
                w=G_D(lr[k],alpha[l],batchsize[m],epoch[n],x_valid,y_valid)
                inter=(np.dot(x_valid,w)-y_valid)**2 
                fmse=np.sum(inter)/(2*x_valid.shape[0])+(alpha[l]/2)*np.dot(w.T,w)
                
                if fmse>0:   # avoid nan values in dictionary                  
                    Hyperparams.update({fmse:[lr[k],alpha[l],batchsize[m],epoch[n]]}) #updates dictionary having MSE as key and hyperparameters as values
hyperparameters=Hyperparams[min(Hyperparams)]
lr_t=hyperparameters[0]
alpha_t=hyperparameters[1]
batchsize_t=hyperparameters[2]
epoch_t=hyperparameters[3]


# In[68]:


w=G_D(lr_t,alpha_t,batchsize_t,epoch_t,x_train,y_train)
lr_t,alpha_t,batchsize_t,epoch_t=[0.0001,0.0005,10,750]
a=np.sum((np.dot(x_te,w)-y_te)**2)/(2*x_te.shape[0])
print("Test MSE (unregularized): ",a) 


# In[66]:


hyperparameters


# In[ ]:




