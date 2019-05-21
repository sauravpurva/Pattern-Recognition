#Assignment-1
#Group-1
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy as np
import math

p=np.array((70,90,80))
q=np.array((40,6,6))
r=np.array((10,20,30))
s=np.array((32,43,55))
t=np.array((70,60,40))
x=np.array((25,20,40))

dist1=distance.euclidean(p,x)
dist2=distance.euclidean(q,x)
dist3=distance.euclidean(r,x)
dist4=distance.euclidean(s,x)
dist5=distance.euclidean(t,x)

dist=[dist1,dist2,dist3,dist4,dist5]
print(dist)
minimum=min(dist1,dist2,dist3,dist4,dist5)
print("Minimum= ",str(minimum))

x=[70,40,10,32,70]
y=[90,6,20,43,60]
z=[80,6,30,55,40]

fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z,c='r')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()




