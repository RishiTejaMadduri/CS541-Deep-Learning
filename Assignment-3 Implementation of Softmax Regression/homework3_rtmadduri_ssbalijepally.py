#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from sklearn import metrics


# In[17]:


x_train=appending("mnist_train_images.npy")
y_train=np.load("mnist_train_labels.npy")
x_te=appending("mnist_test_images.npy")
y_te=np.load("mnist_test_labels.npy")
x_valid=appending("mnist_validation_images.npy")
y_valid=np.load("mnist_validation_labels.npy")

xr,xc=x_te.shape
yr,yc=y_te.shape
print("X train shape",x_train.shape)
print(yc)





#print(x_te.shape)
#w=np.random.randint(1,size=(w.shape[0]))

#from sklearn.model_selection import train_test_split
#x_train, x_valid, y_train, y_valid = train_test_split(x_tr, y_tr, test_size = 0.20, random_state = 0)

#Mapping the x values to y values
mapp = list(zip(x_train, y_train))
np.random.shuffle(mapp)
#Unzipping the values
x_train,y_train=zip(*mapp)


# In[5]:


def appending(X):
    x_tr=np.reshape(np.load(X),(-1,28*28))
    # np.c_[x_tr,np.ones(x_tr.shape[0])]
    return x_tr


# In[7]:


def softmax(X,weights,bias,predict):
    yinte=np.dot(X,weights)+bias
    yexp=np.exp(yinte)
    ypred=yexp/yexp.sum(axis=1,keepdims=True)
    if predict == True:
        print(predict)
        results=[]
        for each in ypred:
            result=np.zeros(each.shape)
            result[each > np.max(each)] = 1
            results.append(result)
        ypred= results
    return ypred


# In[8]:


from sklearn import metrics
def G_D(lr,alpha,batchsize,epoch,X,Y):
    mini_x=[]
    mini_y=[]
    weights=np.random.random(size=(xc,yc))
    bias=np.random.randn(10)
    print("bias",bias.shape)


    
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
            mini_x[j] = np.array(mini_x[j])
            mini_y[j] = np.array(mini_y[j])
            #print("weights:",().shape)
            ypredict=softmax(mini_x[j],weights,bias,False)
            x = np.dot(mini_x[j].T,(ypredict-mini_y[j]))
            GD= x + alpha * weights
            #loss update
            yce=crossentropy(mini_y[j],ypredict)
            print(i)
            print(yce)
            #accuracy=metrics.accuracy_score(mini_y[j], ypredict)
            weights-=lr*GD
            bias-=lr*(ypredict-mini_y[j]).sum(axis=0)            
    return weights,bias


# In[24]:


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


# In[15]:


#lr=[0.00001]
#alpha=[0.1]
#batchsize=[100]
#epoch=[1000]
lr=[0.01,0.0001,0.00001,0.000001]
alpha=[0.1,0.01,0.001,0.0005]
batchsize=[10,50,200,100]
epoch=[ 50,100,500,250]


# In[12]:


def crossentropy(Y,ypred):
    yydot= -Y * np.log(ypred)
    ycei= np.sum(yydot)/ypred.shape[0]
    return ycei
    


# In[20]:


from sklearn import metrics
def accuracy(y_train,y_pred):
    y_train = np.reshape(y_train,(len(y_train),10))
    y_pred = np.reshape(y_pred,(len(y_pred),10))
    if y_train.shape == y_pred.shape:
        return np.sum(y_train == y_pred)/len(y_train)
    


# In[26]:


lr_t,alpha_t,batchsize_t,epoch_t=[0.0001,0.01,100,500]
W_t,b_t=G_D(lr_t,alpha_t,batchsize_t,epoch_t,x_train,y_train)
y_pred=softmax(x_train,W_t,b_t,True)
y_train = np.reshape(y_train,(len(y_train),10))
a_t=accuracy(y_train,y_pred)
print("Accuracy",a_t)


# In[ ]:




