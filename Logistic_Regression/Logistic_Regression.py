import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

#import data
X_train=pd.read_csv('./X_train.csv',header=None)
Y_train=pd.read_csv('./Y_train.csv',header=None)
Y_train=Y_train.rename(columns = {'1':'y'})

X_test=pd.read_csv('./X_test.csv', header=None)
Y_test=pd.read_csv('./Y_test.csv', header=None)
Y_test=Y_test.rename(columns = {'1':'y'})

X_train_mat=X_train.values
Y_train_mat=Y_train.values
X_test_mat=X_test.values
Y_test_mat=Y_test.values

# use steepest ascent algorithm to optimize  


Y_train=Y_train.replace(0, -1)
Y_test=Y_test.replace(0, -1)

X_train['w0']=1
X_test['w0']=1


X_train_mat=X_train.values
Y_train_mat=Y_train.values
X_test_mat=X_test.values
Y_test_mat=Y_test.values

 
w_t=np.zeros([len(X_train_mat[1]),1])      


cost_array=np.empty(10000)

for t in range(10000):
    eta=(1.0/((10**5)*np.sqrt(t+1)))
    a=Y_train_mat*np.dot(X_train_mat,w_t)
    b=np.array([(Y_train_mat*X_train_mat)])
    c=scipy.special.expit(a)
    d=(1-c)*b
    d=np.sum(d, axis=1)
    w_t+=eta*d.transpose()
    cost=np.log(10**-15+c)
    cost=np.sum(cost)
    cost_array[t]=cost
 
# plot costfunction a     
plt.plot(range(10000), cost_array, 'ko')
plt.ylabel('Cost Function', fontsize=18)
plt.xlabel('# of Iterations', fontsize=18)
plt.show()


# use newtons method to optimize

w_t=np.zeros([len(X_train_mat[1]),1])      

cost_array=np.empty(100)


for t in range(100):
    #add=0
    eta=(1.0/np.sqrt(t+1))
    a=Y_train_mat*np.dot(X_train_mat,w_t)
    b=np.array([(Y_train_mat*X_train_mat)])
    c=scipy.special.expit(a)
    d=(1-c)*b
    d=np.sum(d, axis=1)
    d=d.transpose()
    hess=np.zeros([len(X_train_mat[1]),len(X_train_mat[1])]) 
    for i in range(len(X_train_mat)):
        aa=np.dot(X_train_mat[i],w_t)
        bb=(np.array([X_train_mat[i]])).transpose()
        cc=(np.array([X_train_mat[i]]))
        ee=np.dot(bb,cc)
        ff=scipy.special.expit(aa)*(1-scipy.special.expit(aa))*ee
        hess+=-ff
    w_t+=-eta*np.dot(np.linalg.inv(hess),d)
    cost=np.log(10**-15+c)
    cost=np.sum(cost)
    cost_array[t]=cost
              
predict=(np.dot(X_test_mat,w_t)>0)
acc=sum(predict==Y_test_mat)/93
      
# plot objective function across iterations               
plt.plot(range(100), cost_array, 'ko')
plt.ylabel('Objective Function', fontsize=18)
plt.xlabel('# of Iterations', fontsize=18)
plt.figtext(.25, 0.025, 'Predictive Accuracy: 0.914', fontsize=12)
plt.show()
  