#Assignment-2
#GROUP-1
import math
import numpy as np
import pandas as pd
import itertools as it
from matplotlib import pyplot as plt

from sklearn.datasets import load_iris
iris=load_iris()

alpha=pd.DataFrame(iris.data,columns=iris.feature_names)
beta=pd.DataFrame(iris.target)
# print(alpha)
# print(beta)
prior_probability=1/3

w1_initial=[]
for i in range(0,50):
	k1=iris.data[i]
	k1=k1.tolist()
	w1_initial.append(k1)

w2_initial=[]
for i in range(50,100):
	k2=iris.data[i]
	k2=k2.tolist()
	w2_initial.append(k2)

w3_initial=[]
for i in range(100,150):
	k3=iris.data[i]
	k3=k3.tolist()
	w3_initial.append(k3)


#Classw1

a_w1_initial=0
for i in range(10,50):
	a_w1_initial=a_w1_initial+iris.data[i]

mean_septal_length_w1=a_w1_initial[0]/40
mean_septal_width_w1=a_w1_initial[1]/40
mean_petal_length_w1=a_w1_initial[2]/40
mean_petal_width_w1=a_w1_initial[3]/40

mean_vector_w1=[mean_septal_length_w1,mean_septal_width_w1,mean_petal_length_w1,mean_petal_width_w1]

zero_mean_mat_w1=[] 
for i in range(10,50):
	a_w1=[]
	a_w1=iris.data[i]-mean_vector_w1
	a_w1=a_w1.tolist()
	zero_mean_mat_w1.append(a_w1)

mat_transpose_w1=np.transpose(zero_mean_mat_w1)

covariance_w1=1/40*np.dot(mat_transpose_w1,zero_mean_mat_w1)
covariance_mode_w1=np.linalg.det(covariance_w1)
covariance_inverse_w1=np.linalg.inv(covariance_w1)

def normal_density_w1(ly_w1):

	mat1_w1=np.array(ly_w1)-np.array(mean_vector_w1)
	mat1_transpose_w1=np.transpose(mat1_w1)

	dot1_w1=np.dot(mat1_w1,covariance_inverse_w1)
	dot2_w1=np.dot(dot1_w1,mat1_transpose_w1)
	intermediate1_w1=(2*22/7)**(-2)
	intermediate2_w1=(covariance_mode_w1)**(-1/2)
	x=math.e
	intermediate3_w1=x**(-1/2*(dot2_w1))
	ND=intermediate1_w1*intermediate2_w1*intermediate3_w1
	return ND


#Class w2
a_w2_initial=0
for i in range(60,100):
	a_w2_initial=a_w2_initial+iris.data[i]

mean_septal_length_w2=a_w2_initial[0]/40
mean_septal_width_w2=a_w2_initial[1]/40
mean_petal_length_w2=a_w2_initial[2]/40
mean_petal_width_w2=a_w2_initial[3]/40

mean_vector_w2=[mean_septal_length_w2,mean_septal_width_w2,mean_petal_length_w2,mean_petal_width_w2]

zero_mean_mat_w2=[] 
for i in range(10,50):
	a_w2=[]
	a_w2=iris.data[i]-mean_vector_w2
	a_w2=a_w2.tolist()
	zero_mean_mat_w2.append(a_w2)

mat_transpose_w2=np.transpose(zero_mean_mat_w2)

covariance_w2=1/40*np.dot(mat_transpose_w2,zero_mean_mat_w2)
covariance_mode_w2=np.linalg.det(covariance_w2)
covariance_inverse_w2=np.linalg.inv(covariance_w2)

def normal_density_w2(ly_w2):

	mat1_w2=np.array(ly_w2)-np.array(mean_vector_w2)
	mat1_transpose_w2=np.transpose(mat1_w2)

	dot1_w2=np.dot(mat1_w2,covariance_inverse_w2)
	dot2_w2=np.dot(dot1_w2,mat1_transpose_w2)
	intermediate1_w2=(2*22/7)**(-2)
	intermediate2_w2=(covariance_mode_w2)**(-1/2)
	intermediate3_w2=math.e**(-1/2*(dot2_w2))
	ND=intermediate1_w2*intermediate2_w2*intermediate3_w2
	return ND




#Class w3
a_w3_initial=0
for i in range(110,150):
	a_w3_initial=a_w3_initial+iris.data[i]

mean_septal_length_w3=a_w3_initial[0]/40
mean_septal_width_w3=a_w3_initial[1]/40
mean_petal_length_w3=a_w3_initial[2]/40
mean_petal_width_w3=a_w3_initial[3]/40

mean_vector_w3=[mean_septal_length_w3,mean_septal_width_w3,mean_petal_length_w3,mean_petal_width_w3]

zero_mean_mat_w3=[] 
for i in range(10,50):
	a_w3=[]
	a_w3=iris.data[i]-mean_vector_w3
	a_w3=a_w3.tolist()
	zero_mean_mat_w3.append(a_w3)

mat_transpose_w3=np.transpose(zero_mean_mat_w3)

covariance_w3=1/40*np.dot(mat_transpose_w3,zero_mean_mat_w3)
covariance_mode_w3=np.linalg.det(covariance_w3)
covariance_inverse_w3=np.linalg.inv(covariance_w3)

def normal_density_w3(ly_w3):

	mat1_w3=np.array(ly_w3)-np.array(mean_vector_w3)
	mat1_transpose_w3=np.transpose(mat1_w3)

	dot1_w3=np.dot(mat1_w3,covariance_inverse_w3)
	dot2_w3=np.dot(dot1_w3,mat1_transpose_w3)
	intermediate1_w3=(2*22/7)**(-2)
	intermediate2_w3=(covariance_mode_w3)**(-1/2)
	intermediate3_w3=math.e**(-1/2*(dot2_w3))
	ND=intermediate1_w3*intermediate2_w3*intermediate3_w3
	return ND

#CHECKING FOR EVERY ITERATION

w1_final=[]
w2_final=[]
w3_final=[]

for i in range(0,40):
	w1_final.append(w1_initial[i+10])
	w2_final.append(w2_initial[i+10])
	w3_final.append(w3_initial[i+10])

for i in it.chain(range(0,10),range(50,60),range(100,110)):
	A=iris.data[i]
	a=normal_density_w1(A)
	b=normal_density_w2(A)
	c=normal_density_w3(A)
	maximun=max(a,b,c)

	if maximun==a:
		A=A.tolist()
		w1_final.append(A)

	elif maximun==b:
		A=A.tolist()
		w2_final.append(A)

	elif maximun==c:
		A=A.tolist()
		w3_final.append(A)

correct1=0
for i in range(len(w1_initial)):
	for j in range(len(w1_final)):
		if w1_initial[i]==w1_final[j]:
			correct1=correct1+1
		else:
			continue
x1=correct1/(50)
print("Class1Accuracy = ",x1)


correct2=0
for i in range(len(w2_initial)):
	for j in range(len(w2_final)):
		if w2_initial[i]==w2_final[j]:
			correct2=correct2+1
		else:
			continue
x2=correct2/(50)
print("Class2Accuracy = ",x2)

correct3=0
for i in range(len(w3_initial)):
	for j in range(len(w3_final)):
		if w3_initial[i]==w3_final[j]:
			correct3=correct3+1
		else:
			continue
x3=correct3/(50)
print("Class3Accuracy = " ,x3)

w_final=w1_final+w2_final+w3_final


accuracy_score=(x1+x2+x3)/3
print("TotalAccuracy = ",accuracy_score)
plt.scatter(iris.data,w_final,color='red')
plt.title('Accuracy after bayes classification is {}'.format(accuracy_score))
# plt.title('Accuracy of class-1 {}'.format(x1))
# plt.title('Accuracy of class-2 {}'.format(x2))
# plt.title('Accuracy of class-3 {}'.format(x3))
plt.xlabel('Original data')
plt.ylabel('After clustring')
plt.show()





