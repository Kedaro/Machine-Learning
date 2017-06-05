import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Load Data
data_df = pd.read_csv("./nyt_data.csv", header=None)
data=data_df.as_matrix()


vocab_df = pd.read_csv("./nyt_vocab.csv", header=None)
vocab=vocab_df.as_matrix()

#Set rank=number of topics

rank=25

#Initialize W and H (matrix factors) with random values 
np.random.seed(1)

x=np.zeros((len(vocab),len(data)))
W=np.random.uniform(1,2,(len(vocab),rank))
H=np.random.uniform(1,2,(rank,len(data)))

#Reindex data so start at 0
for i in range(len(data)):
    row=data[i]
    row=row[~np.isnan(row)]
    j=0
    while j<len(row):
        ind=int(row[j]-1)
        x[ind,i]=int(row[j+1])
        j+=2

#Add miniscule value to data to avoid issues with 0 vlaues        
x=x+1e-6

#Initialize divergence array (Loss Function)
div=[]


#Calculate W and H and iterate 100 times
for ii in range(100):    
    W_t=W.T
    row_sum=sum(W_t)
    W_t=W_t/row_sum
    
    A=np.dot(W,H)
    B=np.divide(x,A)
    C=np.dot(W_t,B)
    H=H*C
    
    H_t=H.T
    col_sum=sum(H_t)
    H_t=H_t/col_sum
    new_new_sum=sum(H_t)
    
    A1=np.dot(W,H)
    B1=np.divide(x,A1)
    C1=np.dot(B1,H_t)
    
    W=W*C1
    
    div_it=x*np.log(np.dot(W,H))-np.dot(W,H)
    div_it=-1*(np.sum(div_it))
    div.append(div_it)
    print(div_it)
       


#Plot divergence penalty over 100 iterations    
plt.plot(range(1,100),div[1:100])
plt.xlabel('Iteration', fontsize=18)
plt.ylabel('Divergence Penalty', fontsize=16)
plt.show()


#Generate list of words for each topic and their frequency

#Noramlize frequency array of words
W_words=W/sum(W)

#Sort words base on frequncy in topic
W_vals=-1*np.sort(-W_words, axis=0)
W_inds=np.argsort(-W_words, axis=0)

#Store 10 most frequent words for each topic
W_ind=W_inds[:10,:]
W_val=W_vals[:10,:]


#Create list of of list of 10 most frequent words accross 25 topics and their corresponding frequency
wordlist=np.zeros((10,25))
W_ind1=W_ind+1
listoflist=[]
for j in range(25):
    wordlist=vocab[W_ind[:,j]]
    listoflist.append(wordlist)
    listoflist.append(W_val[:,j])


