#Combined or Group Submission of Soumya Balijepally (ssbalijepally) and Rishi Teja Madduri (rtmadduri) #

import numpy as np

A=np.array([(1,2,3),(4,5,6),(3,1,2)])
B=np.array([(3,2,1),(6,5,4),(9,8,7)])
C=np.array([(1,3,2),(4,6,5),(7,9,8)])
x=np.array([[3],[2],[3]])
y=np.array([[3],[2],[1]])
z=np.ones((3, 1))
alpha=7
c=2
d=5
i=1
j=2
k=2
m=2
s=4

def problem_a (A, B):    
  return A + B
#print(problem_a (A, B))

def problem_b (A, B, C): 
  X=np.dot(A,B)  
  return (X-C)
#print(problem_b (A, B, C))

def problem_c (A, B, C):
  a=np.multiply(A,B)
  c=np.transpose(C) 
  return a + c
#print(problem_c (A, B, C))

def problem_d (x, y): 
  x= np.transpose(x) 
  return np.dot(x,y)
#print(problem_d (x, y))

def problem_e (A): 
  d=A.shape  
  return np.zeros(d)
#print(problem_e (A))

def problem_f (A, x):    
  return np.linalg.solve(A, x)
#print(problem_f (A, x))

def problem_g (A, x):
  A = A.T
  x =x.T
  return np.linalg.solve(A,x.T)
#print(problem_g (A, x))

def problem_h (A, alpha):
  w=len(A)
  r=len(A[0]) 
  I=np.eye(r,w)
  alpha=np.multiply(alpha,I)  
  return A+alpha
#print(problem_g (A, x))

def problem_i (A, i, j):
  return A.item((i, j))
#print(problem_i (A, i, j))

def problem_j (A, i):  
  asum= A[i,0:A.shape[1]:2] 
  return np.sum(asum) 
#print(problem_j (A, i))

def problem_k (A, c, d): 
  ac=A[np.nonzero(A >= c)]
  ad=ac[np.nonzero(ac <= d)]
  meaan=np.mean(ad)
  return meaan
#print(problem_k (A, c, d))

def problem_l (A, k):
  w,v=np.linalg.eig(A)
  sortw=-np.sort(-w) 
  sortw=sortw.tolist()
  list1=[]
  for i in range(k): 
    list1.append (sortw[i]) 
  list2=[]
  wlist=w.tolist()
  for i in range(0,k):
    #print(sortw.index(wlist[i]))
    list2.append(sortw.index(wlist[i]))
  list3=[]
  for i in range(0,k):
    #print((v[list2[i]]))
    list3.append((v[list2[i]]))
  list3=np.asarray(list3)
  list3=np.transpose(list3)
  return list3
#print(problem_l (A, k))

def problem_m (x, k, m, s):

  mz=np.multiply(m,z)
  meann=x+mz
  meann=np.transpose(meann)
  meann=tuple(meann.reshape(1, -1)[0])
  n=np.size(x)
  IdentityM=np.eye(n,n)
  covariance= np.multiply(s,IdentityM)
  GND=np.random.multivariate_normal(meann,covariance,(k))
  return GND
#print(problem_m (x, k, m, s))

def problem_n (A): 
  return np.random.permutation(A)
#print(problem_n (A))

def linear_regression (X_tr, y_tr):
  str=np.dot(X_tr , X_tr.T)
  w = np.linalg.solve(str, np.dot(X_tr, y_tr)
  return w

import numpy as np

def linear_regression (X_tr, y_tr):
    sumtr=np.dot(X_tr , X_tr.T)
    w = np.linalg.solve(sumtr, np.dot(X_tr, y_tr))
    return w 

def train_age_regressor ():
# Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1,48*48,))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1,48*48,))
    yte = np.load("age_regression_yte.npy")
    
    w = linear_regression(X_tr.T, ytr)

# Report fMSE training
    FMSE_Tr_Vec=np.multiply(np.subtract(np.dot(X_tr,w),ytr),np.subtract(np.dot(X_tr,w),ytr))
    FMSE_TR=1/2*np.mean(FMSE_Tr_Vec)
    print(FMSE_TR)

# Report fMSE
    FMSE_TE_Vec=np.multiply(np.subtract(np.dot(X_te,w),yte),np.subtract(np.dot(X_te,w),yte))
    FMSE_TE=1/2*np.mean(FMSE_TE_Vec)
    print(FMSE_TE)

train_age_regressor ()

