import numpy as np
import matplotlib.pyplot as plt


#determine the number of classes
k=5

#Generate data using gaussian mixture
np.random.seed(0)
#choose distribiution of data across gaussian
options=np.array([0,1,2])
p=np.array([.2,.5,.3])
num=np.random.choice(options, size=500, p=p)

num_a=np.sum(num==0)
num_b=np.sum(num==1)
num_c=np.sum(num==2)

#generate 3 normal distrutions with differnt mean values
a=np.random.multivariate_normal([0,0],[[1,0],[0,1]],num_a)
b=np.random.multivariate_normal([3,0],[[1,0],[0,1]],num_b)
c=np.random.multivariate_normal([0,3],[[1,0],[0,1]],num_c)

#mix generated data
data=np.concatenate((a,b,c))

#initalize classification array
classify=np.zeros(len(data))

#inialize cluster centroids as random
#means=np.random.rand(k,2)
#inialize distance array
#dist=np.zeros(k)

#initalize loss array
loss=np.zeros((4,20))


#iterate over number of classes
for jj in range(2,k+1):
    #inialize cluster centroids as random
    means=np.random.rand(jj,2)
    #inialize distance array
    dist=np.zeros(jj)
    
    #iterate 20 times
    for ii in range(20):
        
        #classify data based on proximity to centroid
        for i in range(len(data)):
            for j in range(jj):
                aa=np.linalg.norm(data[i]-means[j])**2
                dist[j]=aa
            classify[i]=np.argmin(dist)
        
        #redefine centroids as average of all data in class
        for i in range(jj):
            aaa=sum(data[classify==i])
            bbb=(1/sum(classify==i))
            ccc=bbb*aaa
            means[i]=ccc
        

#plot data, color corresponding to assigned class
x=data[:,0]
y=data[:,1]
plt.scatter(x[classify==0],y[classify==0], color='r',marker='^')
plt.scatter(x[classify==1],y[classify==1], color='g',marker='s')
plt.scatter(x[classify==2],y[classify==2], color='b',marker='x')
#plt.scatter(x[classify==3],y[classify==3], color='y',marker='o') 
#plt.scatter(x[classify==4],y[classify==4], color='c',marker='x')
plt.show()