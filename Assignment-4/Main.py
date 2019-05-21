#Assignment-4
#Group-1
import numpy as np
import pandas as pd
import math
from operator import add
from matplotlib import pyplot as plt

learning_rate=1 #can be set variable

w2=[[0.5,3.0],[1,3.0],[0.5,2.5],[1,2.5],[1.5,2.5]] #class w2
w1=[[4.5,1],[5,1],[4.5,0.5],[5.5,0.5]] #classs w1

we=w1+w2 #
# w2=[[0,0]]
# w1=[[0,1],[1,0],[1,1]]

neg=[-1,-1,-1]
neg=np.transpose(neg)
for i in range(0,5):
	w2[i].append(1)
for i in range(0,4):
	w1[i].append(1)

x=np.array(w2)
y=-x
w2=list(y)
w3=w2
# print(w3)

mat=[0,0,0]

def non_zero(list1,list2): #To check if value if >0,<0 or =0
	list1=np.transpose(list1)
	val=np.dot(list1,list2)
	return val

def update(list1,list2): #Update the iteration by rule
	list3=list(map(add,list1,learning_rate*list2))
	return list3

w=w2+w1
# print(w)
counter=0
i=0
# it=0
while counter<9: #unless all the values are classifies properly 	
	value=non_zero(mat,w[i])
	# print(value)
	if value>0:
		counter=counter+1
		i=(i+1)%9

		continue
	elif value<=0:
		mat=update(mat,w[i])
		# print(mat)
		# it=it+1
		counter=0
		i=(i+1)%9

print('Value of A,B and C in Ax1+Bx2+C=0 is {},{},{}'.format(mat[0],mat[1],mat[2]))
# print(it)

xco=[]
yco=[]
x = np.arange(-10, 10, 0.01)

for i in x:
    #a2*x1 + a1*x2 +a0=0
    a2=mat[0]
    a1=mat[1]
    a0=mat[2]
    y1=-(a0+(a2*i))/a1
    xco.append(i)
    yco.append(y1)


plt.plot(xco, yco,color='black')

d=len(we)

for i in range(0,4):
    class1=plt.scatter(we[i][0], we[i][1], color="green",marker='o')
for i in range(4,d):
    class2=plt.scatter(we[i][0], we[i][1], color="red",marker='*')
for i in range(0,5):
	class3=plt.scatter(w3[i][0],w3[i][1], color="black", marker="*")

plt.title('Perceptron Classification')
plt.xlabel("x1")
plt.ylabel("x2")

plt.show()

