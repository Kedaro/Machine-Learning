import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt



#load data
x_train_df = pd.read_csv("./boosting/X_train.csv", header=None)
y_train_df = pd.read_csv("./boosting/y_train.csv", header=None)
x_test_df = pd.read_csv("./boosting/X_test.csv", header=None)
y_test_df = pd.read_csv("./boosting/y_test.csv", header=None)
x_train=x_train_df.values
y_train=y_train_df.values

y_train=y_train[:,0]
x_test=x_test_df.values
y_test=y_test_df.values
y_test=y_test[:,0]

#add column of ones
ones_test=np.ones((len(x_test),1))
ones_train=np.ones((len(x_train),1))

x_train=np.concatenate((x_train,ones_train), axis=1)
x_test=np.concatenate((x_test,ones_test), axis=1)

#variables
T=1500
boot_ind=np.zeros((T,len(x_train)))
e_t=np.zeros((T))
alpha=np.zeros((T))

train_err=np.zeros((T))
test_err=np.zeros((T))

boost_test_val=np.zeros(len(x_test))
boost_train_val=np.zeros(len(x_train))

hist=np.zeros((len(x_train)))

wts=np.zeros(len(x_train))
wts.fill(1/len(x_train))

length_train=len(x_train)

ubsum=0
up_bound=np.zeros((T))


np.random.seed(1989)
for ii in range(T):
    
    #take bootstrap
    boot=np.random.choice(length_train, length_train, replace=True, p=wts)
    x_train_boot=x_train[boot]
    y_train_boot=y_train[boot]
    boot_ind[ii]=boot
            
    #calculate LS        
    w=np.dot(x_train_boot.T,x_train_boot)
    w=np.linalg.inv(w)
    w=np.dot(w,np.dot(x_train_boot.T,y_train_boot))
    
    #calculate epsilon
    f_x=np.sign(np.dot(x_train,w))
    wrong_1= f_x!=y_train
    e_t[ii]=np.sum(wts[wrong_1])
    if e_t[ii]>.5:
        w=w*-1
        f_x=np.sign(np.dot(x_train,w))
        e_t[ii]=np.sum(wts[f_x!=y_train])
    f_x_test=np.sign(np.dot(x_test,w))
        
    #calculate alpha    
    alpha[ii]=.5*np.log((1-e_t[ii])/e_t[ii])

    #predict values
    boost_train_val+=alpha[ii]*f_x    
    boost_test_val+=alpha[ii]*f_x_test

    boost_train=np.sign(boost_train_val)
    boost_test=np.sign(boost_test_val)
    
    #calculate train/test error
    train_err[ii]=np.sum(boost_train!=y_train)/len(y_train)
    test_err[ii]=np.sum(boost_test!=y_test)/len(y_test)
    
    #calculate upper bound on error
    ubsum+=-2*((.5-e_t[ii])**2)
    ub=np.exp(ubsum)
    up_bound[ii]=ub
    
    print(ii) 
    
    #update wts
    for i in range(len(wts)):
        wts[i]=wts[i]*np.exp(-alpha[ii]*y_train[i]*f_x[i])  
    wts=wts/np.sum(wts)

  
#generated plot
plt.plot(range(T),train_err,label="Training Error")
plt.plot(range(T),test_err,label="Testing Error")
plt.suptitle('Part 2a', fontsize=20)
plt.xlabel('Number of Iterations', fontsize=18)
plt.ylabel('Error', fontsize=16)
plt.legend(bbox_to_anchor=(.75, .95), loc=2, borderaxespad=0.)
plt.show()