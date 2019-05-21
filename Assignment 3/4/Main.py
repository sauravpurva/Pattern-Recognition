#Assignment-3.4
#Group-1
import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import math

class1=[]
class2=[]
for index in range(1,13):
    rgbImg1 = io.imread('i'+str(index)+'.jpg')
    rgbImg2 = io.imread('k'+str(index)+'.JPG')
    grayImg1 = img_as_ubyte(color.rgb2gray(rgbImg1))
    grayImg2 = img_as_ubyte(color.rgb2gray(rgbImg2))

    distances = [1]
    angles = [0]
    properties = ['energy', 'homogeneity','contrast']

    glcm1 = greycomatrix(grayImg1,
                        distances=distances,
                        angles=angles,
                        symmetric=True,
                        normed=True)
    glcm2 = greycomatrix(grayImg2,
                        distances=distances,
                        angles=angles,
                        symmetric=True,
                        normed=True)

    class1.append(np.hstack([greycoprops(glcm1, prop).ravel() for prop in properties]))
    class2.append(np.hstack([greycoprops(glcm2, prop).ravel() for prop in properties]))

print(class1)
print(class2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for point in class1:
    print(point[0],point[1],point[2])
    ax.scatter(point[0],point[1],point[2],color="r",marker='*')

for point in class2:
    print(point[0],point[1],point[2])
    ax.scatter(point[0],point[1],point[2],color='b',marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

mean1,mean2=np.mean(class1,axis=0),np.mean(class2,axis=0)
#print(mean1,mean2)

z1=[np.array(class1)[:,0]-mean1[0],np.array(class1)[:,1]-mean1[1],np.array(class1)[:,2]-mean1[2]]
z2=[np.array(class2)[:,0]-mean2[0],np.array(class2)[:,1]-mean2[1],np.array(class2)[:,2]-mean2[2]]



n=12
cov1 = [i/n for i in (np.dot(np.array(z1),np.array(z1).T))]
cov2 = [i/n for i in (np.dot(np.array(z2),np.array(z2).T))]


print(cov1,cov2)
covInv1=np.linalg.inv(cov1)
covInv2=np.linalg.inv(cov2)

print(covInv1,covInv2)

a1=[i/(-2) for i in covInv1]
a2=[i/(-2) for i in covInv2]

b1= np.dot(covInv1,mean1)
b2= np.dot(covInv2,mean2)

c1=np.dot(np.dot(mean1.T,covInv1),mean1)/(-2) - (np.log(np.linalg.det(cov1))*0.5)
c2=np.dot(np.dot(mean2.T,covInv2),mean2)/(-2) - (np.log(np.linalg.det(cov2))*0.5)

A=np.subtract(np.array(a1),np.array(a2))

x1x1=A[0][0]
x1x2=A[0][1]+A[1][0]
x1x3=A[0][2]+A[2][0]
x2x2=A[1][1]
x2x3=A[1][2]+A[2][1]
x3x3=A[2][2]

B=np.subtract(np.array(b1),np.array(b2))
#print(b1)
#print(b2)
#print(B)
x1=B[0]
x2=B[1]
x3=B[2]
c0=c1-c2

print(x1x1,x1x2,x1x3,x2x2,x2x3,x3x3,x1,x2,x3,c0)

x=np.arange(0,0.9,0.01)
y=np.arange(0,0.9,0.01)
xx, yy = np.meshgrid(x, y)
print(xx,yy)

xx1=np.hstack(xx)
yy1=np.hstack(yy)
print(xx1.T,yy1.T)

#z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

a=np.full(np.shape(xx),x3x3)

b= (i * x2x3 for i in yy) + (i*x1x3 for i in xx)+ np.full(np.shape(xx), x3)

print(b.shape())
c= ((i*i) * x2x3 for i in xx) + ((j*j) * x2x2 for j in yy) + (i* x1x2 for i in np.dot(xx,yy))+(i * x1 for i in xx)+(j * x2 for j in yy)+np.full(np.shape(xx),c0)

des= np.dot(b,b)- (i*4 for i in np.dot(a,c))

z= -b + np.divide(np.sqrt(des),(i*2 for i in a))

for i in xx1:
    for j in yy1:
        a=x3x3

        b=(x2x3*j)+(x1x3*i)+x3
        c=x1x1*i*i+x2x2*j*j+x1x2*i*j+x1*i+x2*j+c0

        #print(a,b,c)
        des=(b*b)-(4*a*c)
        if(des>0):
            #print(math.sqrt(des))
            r1=(-b+math.sqrt(des))/(2*a)
            pos[xx1.index(i)][yy1.index(j)]=r1

for i, j in zip(range(xx1.T), range(yy1.T)):
    a=x3x3
    b=(x2x3*j)+(x1x3*i)+x3
    c=x1x1*i*i+x2x2*j*j+x1x2*i*j+x1*i+x2*j+c0

    #print(a,b,c)
    des=(b*b)-(4*a*c)
    if(des.all()>0):
        #print(math.sqrt(des))
        r1=(-b+math.sqrt(des))/(2*a)
        posz.append(r1)


plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, np.array(posz), alpha=0.2)