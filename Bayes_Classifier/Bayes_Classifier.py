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


#calculate maximum likelyhood of y_training (average)
pi_ml=Y_train_mat.sum(0)
pi_ml=pi_ml/len(Y_train_mat)




#seperate bernoulli variables for both positive and negative cases
x_0=X_train_mat*(Y_train_mat==0)
x_1=X_train_mat*(Y_train_mat==1)
                    
bern_x_0=x_0[1776:,:54]
bern_x_1=x_1[:1776,:54]


#seperate pareto variables for both positive and negative cases

par_x_0=x_0[1776:,54:]
par_x_1=x_1[:1776,54:]  

      

#calculate ML vector bernoulli
bern_ml_0=np.sum(bern_x_0, axis=0)/sum(Y_train_mat==0)
bern_ml_1=np.sum(bern_x_1, axis=0)/sum(Y_train_mat==1)


#calculate ML vector pareto 
par_ml_0=np.log(par_x_0)
par_ml_0=np.sum(par_ml_0, axis=0)
par_ml_0=sum(Y_train_mat==0)/par_ml_0
            
par_ml_1=np.log(par_x_1)
par_ml_1=np.sum(par_ml_1, axis=0)
par_ml_1=sum(Y_train_mat==1)/par_ml_1
           

#predict test values
bern_test=X_test_mat[:,0:54]
par_test=X_test_mat[:,54:]
p_bern_0=np.ones(len(X_test_mat))
p_bern_1=np.ones(len(X_test_mat))

p_par_0=np.ones(len(X_test_mat))
p_par_1=np.ones(len(X_test_mat))


#caclulate bern probabilities
for i in range(len(bern_test)):
    for j in range(len(bern_test[1])):
        p_bern_0[i]=p_bern_0[i]*(bern_ml_0[j]**bern_test[i,j]\
                *((1-bern_ml_0[j])**(1-bern_test[i,j])))

for i in range(len(bern_test)):
    for j in range(len(bern_test[1])):
        p_bern_1[i]=p_bern_1[i]*(bern_ml_1[j]**bern_test[i,j]\
                *((1-bern_ml_1[j])**(1-bern_test[i,j])))

#calculate pareto probabilities
p_par_0=np.ones(len(X_test))
p_par_1=np.ones(len(X_test))

for i in range(len(par_test)):
    for j in range(len(par_test[1])):
        p_par_0[i]=p_par_0[i]*(par_ml_0[j]*\
               (par_test[i,j]**(-(par_ml_0[j]))))

for i in range(len(par_test)):
    for j in range(len(par_test[1])):
        p_par_1[i]=p_par_1[i]*(par_ml_1[j]*\
               (par_test[i,j]**(-(par_ml_1[j]))))
      
#predict class of data
p_0=np.ones(len(X_test))
p_1=np.ones(len(X_test))
predict=np.zeros(len(X_test))
for i in range(len(X_test)):
    p_0[i]=(1-pi_ml)*p_bern_0[i]*p_par_0[i]
    p_1[i]=(pi_ml)*p_bern_1[i]*p_par_1[i]
    if p_0[i]>p_1[i]:
        predict[i]=0
    else:
        predict[i]=1

#create truth table to evaluate predictions
table=np.identity(2)
table[0,0]=sum(np.logical_and(predict==0,Y_test_mat[:,0]==0))
table[1,1]=sum(np.logical_and(predict==1,Y_test_mat[:,0]==1))
table[0,1]=sum(np.logical_and(predict==1,Y_test_mat[:,0]==0))
table[1,0]=sum(np.logical_and(predict==0,Y_test_mat[:,0]==1))

table_df=pd.DataFrame(table, columns=['y=0', 'y=1'], index = ['predict=0', 'predict=1'])

accuracy=(table[0,0]+table[1,1])/len(predict)
print(accuracy)

 
# create stem plot showing bernoulli parameters for dimensions
# parameter value corresponds to inidcator of SPAM

x=np.linspace(0, 54, 54)
ml_0=np.concatenate((bern_ml_0, par_ml_0))
ml_1=np.concatenate((bern_ml_1, par_ml_1))
plt.subplot(211)
plt.title('Bernoulli Paramaters y=0',fontsize=25)
plt.stem(x, bern_ml_0, 'b-.')

plt.subplot(212)
plt.title('Bernoulli Paramaters y=1',fontsize=25)        
plt.stem(x, bern_ml_1, 'r-.')
plt.show()


# use K-NN to try and predict data

# utilize l1 norm

l1_norm=np.zeros((len(X_test_mat),len(X_train_mat)))


for i in range(len(X_test_mat)):
    print(i)
    for j in range(len(X_train_mat)):
        l1_norm[i,j]=np.linalg.norm((X_test_mat[i] - X_train_mat[j]), ord=1)
        #l1_norm_2[i,j]=np.sum(np.absolute(np.subtract(X_test_mat[i],X_train_mat[j]))) 

predict=np.zeros((len(X_test_mat),20))

        
for k in range(1,21):
    for i in range(len(X_test_mat)):
        a=np.argsort(l1_norm[i])[:k]
        predict[i,k-1]=round(np.sum(Y_train_mat[a])/float(k))

# determine accuracy of k-nn approach and compare accross values of k
measure=predict-Y_test_mat
acc=sum(measure==0)/93
       
x=range(1,21)
plt.plot(x, acc)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.ylabel('Accuracy', fontsize=18)
plt.xlabel('# of Nearest Neighbors', fontsize=18)
plt.show()
