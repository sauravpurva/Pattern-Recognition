#Assignment-3.3
#Group-1
import pickle
import converter
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

with open('triangles.pkl', 'rb') as f:
    triangles = pickle.load(f)

with open('circles.pkl', 'rb') as f:
    circles = pickle.load(f)


mean1,mean2=np.mean(triangles,axis=0),np.mean(circles,axis=0)
z1=[np.array(triangles)[:,0]-mean1[0],np.array(triangles)[:,1]-mean1[1]]
z2=[np.array(circles)[:,0]-mean2[0],np.array(circles)[:,1]-mean2[1]]


n=10
cov1 = [i/n for i in (np.dot(np.array(z1),np.array(z1).T))]
cov2 = [i/n for i in (np.dot(np.array(z2),np.array(z2).T))]

#print(cov1,cov2)
covInv1=np.linalg.inv(cov1)
covInv2=np.linalg.inv(cov2)

a1=[i/(-2) for i in covInv1]
a2=[i/(-2) for i in covInv2]

b1= np.dot(covInv1,mean1)
b2= np.dot(covInv2,mean2)

c1=np.dot(np.dot(mean1.T,covInv1),mean1)/(-2) - (np.log(np.linalg.det(cov1))*0.5)
c2=np.dot(np.dot(mean2.T,covInv2),mean2)/(-2) - (np.log(np.linalg.det(cov2))*0.5)

#print(c1,c2)


A=np.subtract(np.array(a1),np.array(a2))

x1x1=A[0][0]
x1x2=A[0][1]+A[1][0]
x2x2=A[1][1]

B=np.subtract(np.array(b1),np.array(b2))
x1=B[0]

x2=B[1]
c0=c1-c2

x = np.arange(1, 2.5, 0.00001)

posx=[]
posy=[]
negx=[]
negy=[]

#0 = c + x2*Y +x1*X + x2x2*Y*Y + x1x2*X*Y + x1x1*X*X

for i in x:
    #0 = c0 + x2*Y +x1*i + x2x2*Y*Y + x1x2*i*Y + x1x1*i*i
    a=x2x2
    b=(x1x2*i)+x2
    c=(x1x1*i*i)+(x1*i)+c0

    #print(a,b,c)
    des=(b*b)-(4*a*c)
    if(des>0):
        #print(math.sqrt(des))
        r1=(-b+math.sqrt(des))/(2*a)
        posx.append(i)
        posy.append(r1)

plt.plot(posx, posy,color='blue')

for i in x[::-1]:
    #0 = c0 + x2*Y +x1*i + x2x2*Y*Y + x1x2*i*Y + x1x1*i*i
    a=x2x2
    b=(x1x2*i)+x2
    c=(x1x1*i*i)+(x1*i)+c0
    #print(a,b,c)
    des=(b*b)-(4*a*c)
    if(des>0):
        print(math.sqrt(des))
        r2=(-b-math.sqrt(des))/(2*a)
        negx.append(i)
        negy.append(r2)

plt.plot(negx, negy,color='blue')

#print(newx)
#print(y)

plt.title('Bayes Classification')
plt.xlabel("x1=colour")
plt.ylabel("x2=size")
#print(triangles)
for i in range(0,10):
    print(triangles[i][0],triangles[i][1])
    tri=plt.scatter(triangles[i][0], triangles[i][1], color="green",marker='o')
utri=plt.scatter(mean1[0],mean1[1],color="yellow",marker='+')
for i in range(0,10):
    print(circles[i][0],circles[i][1])
    cir=plt.scatter(circles[i][0], circles[i][1], color="red",marker='o')
ucir=plt.scatter(mean2[0],mean2[1],color="blue",marker='+')

img1=Image.open('test.png')
test=converter.getColorAndSize(img1)
print(test)
testimage=plt.scatter(test[0],test[1],marker='*')

plt.legend((tri, cir,utri,ucir,testimage),
           ('class 1=green triangles', 'class 2=red circles','class 1 mean','class 2 mean','test image'),
           loc='lower left',
           scatterpoints=1,
           fontsize=8)

plt.show()
