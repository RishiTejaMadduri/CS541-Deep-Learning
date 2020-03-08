import numpy as np
import math

#Loading Datat
X_tr=np.load("mnist_train_images.npy")
y_tr=np.load("mnist_train_labels.npy")
X_te=np.load("mnist_test_images.npy")
y_te=np.load("mnist_test_labels.npy")
X_valid=np.load("mnist_validation_images.npy")
y_valid=np.load("mnist_validation_labels.npy")

#Weight Initialization
def make_weights(num_layers,num_neurons):
	W = {}
	b = {}
	for i in range(1,num_layers+2):
		C=(1/float(num_neurons[i-1]))**0.5
		if i == 1:
			W[i] =  (C)*np.random.randn(784,num_neurons[i-1])
			b[i] = (C)*np.random.randn(1,num_neurons[i-1])
		else:
			W[i] =  (C)*np.random.randn(num_neurons[i-2],num_neurons[i-1])
			b[i] = (C)*np.random.randn(1,num_neurons[i-1])
		print("Shape of W{i} is: {z}\nShape of b{i} is: {b}".format(i=i,z=W[i].shape,b=b[i].shape))
	return W,b	

#Calculating Z
def cal_Z(X_tr,W,B):
    Z=np.add(np.dot(X_tr,W),B)
    return Z

#Function to Compute Softmax
def softmax(z):
    yhat=np.zeros(z.shape)
    for k in range(len(z)):      
        yhat[k]=np.exp(z[k])/np.sum(np.exp(z[k]))
    return yhat
        
#Function to compute ReLu 
def relu(Z):
    Z=np.array(Z)
    Z[Z<=0]=0
    return Z

#Function to compute Log Loss
def log_loss(yhat,y_tr,W,alpha):
	sum_w = 0
	for j in range(1,len(W)+1):
	    for i in range(len(yhat[-1,:])):                
	        W_reg = np.dot(W[j][:-1,i].T,W[j][:-1,i])
	        sum_w = sum_w + W_reg

	Loss = -1/len(y_tr[:,0])*np.sum(y_tr*np.log(yhat))
	Loss_reg = Loss + alpha/2*sum_w
	return Loss_reg 

#Function to Calculate Accuracy over Validation Datasets
def accuracy(X_valid,num_layers,y_valid,W,b):
    yhat,Z = f_prop(X_tr,num_layers,W,b)            
    yhat_boolean=(yhat.argmax(axis=1)==y_tr.argmax(axis=1))
    return (np.count_nonzero(yhat_boolean == True)/float(len(y_tr)))*100     

#Function to Compute derivative of ReLu or Relu Prime
def relu_prime(Z):
    Z=np.array(Z)
    Z[Z<=0]=0
    Z[Z>0]=1
    return Z

#Function to compute Forward Propagation
def f_prop(X_tr,num_layers,W,B):
    Z={}
    Z[0]=X_tr
    for i in range(1,num_layers+2):
        if i==1:
            Z[i]=relu(cal_Z(X_tr,W[i],B[i]))
        else:
            Z[i]=relu(cal_Z(Z[i-1],W[i],B[i]))
    yhat=softmax(Z[num_layers+1])
    return yhat,Z

#Function to Tune Hyperparameters
def hyperparameters(W,b,X_tr,y_tr,X_valid,y_valid):
    L_old=10000
    lr=[0.01,0.0001,0.00001,0.000001]
    alpha=[0.1,0.01,0.001,0.0005]
    batchsize=[5,10,25,100]
    epoch=[50,100,200,500]
    for i in range(len(lr)):
        for j in range(len(lr)):
            for k in range(len(lr)):
                for l in range(len(lr)):
                    
                    W,b=train(W,b,X_tr,y_tr,num_layers,epoch[l],batchsize[k],alpha[j],lr[i])
                    yhat_new=softmax(Z)
                    L_new=log_loss(yhat_new,y_valid,W,alpha)
                    
                    if(L_new<L_old):
                        L_old=L_new
                        epoch_new=epoch
                        mini_batch_size=batchsize
                        alpha_new=alpha
                        lr_new=lr
                        W_new=W
                        print("Validation set loss: {L}".format(L=L_new))


    return W_new, epoch_new, mini_batch_size, alpha_new, lr_new
    

#Training function which computes forward and backward propagation
def train(W,b,Xtr,ytr,num_layers,epoch,mini_batch_size,alpha,lr):

    dL_db = {}
    dL_dw = {}
    for _ in range(epoch):
        for j in range(int(math.floor(len(Xtr[:,-1])/mini_batch_size))):
            
            start = j*mini_batch_size
            stop = start + mini_batch_size

            yhat,Z =f_prop(X_tr[start:stop,:],num_layers,W,b)

            l = log_loss(yhat,y_tr[start:stop,:],W,0.0)


            g_T = (1/len(yhat[:,-1]))*(yhat-y_tr[start:stop,:])
            g_T_c = np.sum(g_T,axis = 0)
            g_T_c = g_T_c.reshape(1,len(g_T_c))

            dL_db[num_layers+1] = g_T_c
            dL_dw[num_layers+1] = np.dot(Z[(num_layers+1)-1].T,g_T)
            
 	
            
            for l in range(num_layers,0,-1):
                g_T =np.dot(g_T,W[l+1].T)*relu_prime(Z[l])
                g_T_c = np.sum(g_T,axis=0)
                g_T_c = g_T_c.reshape(1,len(g_T_c))
                dL_dw[l] = np.dot(Z[l-1].T,g_T)
                dL_db[l] = g_T_c

            for i in range(1,num_layers+2):
            	b[i] -= lr*dL_db[i]
            	W[i] -= lr*dL_dw[i]

            acc = accuracy(Xtr[start:stop,:],num_layers,y_tr[start:stop,:],W,b)
            print(" Accuracy: {a}".format(a = acc))
    
    Loss = 0     #Avoiding overflow
    return W,b

#Initializing Number of Layers and Number of Neurons         
num_layers = 2
num_neurons = [50,40,10]

#Function Calling
W,B = make_weights(num_layers,num_neurons) #Initializing Weights
yhat,Z = f_prop(X_tr,num_layers,W,B) #Calculating Forward Propagation to check its working properly

W, epoch, mini_batch_size, alpha, lr=hyperparameters(W,B,X_tr,y_tr,X_valid,y_valid) #Hyperparameter function

