import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize  # For check_grad, approx_fprime


class RNN:
    def __init__ (self, numHidden, numInput, numOutput):
        self.numHidden = numHidden
        self.numInput = numInput
        self.U = np.random.randn(numHidden, numHidden) * 1e-1
        self.V = np.random.randn(numHidden, numInput) * 1e-1
        self.W = np.random.randn(numHidden) * 1e-1
        # TODO: IMPLEMENT ME

    def backward (self,x, y):
        yhat=[]
        Grad_W=0
        Grad_U=0
        Grad_V=0
        U_initial=np.zeros((1,6))
        V_initial=x[0]
        h=np.zeros((51,6))
        z=np.zeros((50,6))
        
        for i in range(1,51):

            z[i-1,:] = np.dot(h[i-1,:],self.U) + np.dot(x[i-1],self.V).T[0]
            h[i,:] = np.tanh(z[i-1,:])
            yhat.append(np.dot(self.W,h[i,:]))
            Grad_W=Grad_W+np.dot((yhat[i-1]-y[i-1]),h[i,:])
            
#            U_initial=h[i-1,:]+np.dot(U_initial,np.dot(self.U,(np.diag(1-np.square(h[i-1,:])))))
            U_initial=np.dot(U_initial,np.dot(self.U,(np.diag(1-np.square(h[i-1,:])))))
            yht_yt=np.dot(yhat[i-1]-y[i-1],self.W)
            Ft=1-np.square(h[i,:])
            dh_dvec=np.dot(yht_yt,Ft)
            
            Grad_U=Grad_U+np.dot(dh_dvec,U_initial[0])
            
#            V_initial=x[i-1]+np.dot(V_initial, np.dot(self.U,np.diag(1-np.square(h[i-1,:]))))
            V_initial=np.dot(V_initial, np.dot(self.U,np.diag(1-np.square(h[i-1,:]))))
            Grad_V=Grad_V+np.dot(dh_dvec,U_initial[0])

        return Grad_W,Grad_U,Grad_V, h,z
            
        # TODO: IMPLEMENT ME
        pass
            
    def forward (self,x,y,h,z):
        yhat = []
        for i in range(1,51):
            yhat.append(np.dot(h[i].T,self.W))
        cost = np.sum(0.5*np.square(yhat - y))
        print(cost) 
    
    def Grad(self,x,y):
        learning_rate=1e-1
        for i in range(100000):
            Grad_W, Grad_U, Grad_V,h,z=self.backward(x,y)
            (x,y)
            self.W=self.W-(learning_rate)*Grad_W
            self.U=self.U-(learning_rate)*Grad_U
            self.V=self.V-(learning_rate)*Grad_V
            self.forward(x,y,h,z)


# From https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
def generateData ():
    total_series_length = 50
    echo_step = 2  # 2-back task
    #batch_size = 1
    x = np.random.choice(2, total_series_length, p=[0.5, 0.5])
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    y = list(y)
    return (x, y)


if __name__ == "__main__":
    xs, ys = generateData()
    print (xs)
    print (ys)
    numHidden = 6
    numInput = 1
    numTimesteps = len(xs)
    rnn = RNN(numHidden, numInput, 1)
    # TODO: IMPLEMENT ME
    rnn.Grad(np.array(xs),np.array(ys))