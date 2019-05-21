#Assignment-3.1
#Group-1
import numpy as np
import matplotlib.pyplot as plt
import math

def getClass(x,y,x1x1,x1x2,x2x2,x1,x2,c0):
    f= c0 + x2*y +x1*x + x2x2*y*y + x1x2*x*y + x1x1*x*x
    if(f>0):
        print("The point (%d,%d) belongs to class 1"%(x,y))
    elif(f<0):
        print("The point (%d,%d) belongs to class 2"%(x,y))
    else:
        print("The point lies on the decision boundary")


def main():
    w1=[(0,0),(0,1),(2,2),(3,1),(3,2),(3,3)]
    w2=[(6,9),(8,9),(9,8),(9,9),(9,10),(8,11)]

    mean1,mean2=np.mean(w1,axis=0),np.mean(w2,axis=0)
    z1=[np.array(w1)[:,0]-mean1[0],np.array(w1)[:,1]-mean1[1]]
    z2=[np.array(w2)[:,0]-mean2[0],np.array(w2)[:,1]-mean2[1]]

    print(mean1,mean2)

    n=6
    cov1 = [i/n for i in (np.dot(np.array(z1),np.array(z1).T))]
    cov2 = [i/n for i in (np.dot(np.array(z2),np.array(z2).T))]

    covInv1=np.linalg.inv(cov1)
    covInv2=np.linalg.inv(cov2)

    a1=[i/(-2) for i in covInv1]
    a2=[i/(-2) for i in covInv2]

    b1= np.dot(covInv1,mean1)
    b2= np.dot(covInv2,mean2)

    c1=np.dot(np.dot(mean1.T,covInv1),mean1)/(-2) - (np.log(np.linalg.det(cov1))*0.5)
    c2=np.dot(np.dot(mean2.T,covInv2),mean2)/(-2) - (np.log(np.linalg.det(cov2))*0.5)

    A=np.subtract(np.array(a1),np.array(a2))

    x1x1=A[0][0]
    x1x2=A[0][1]+A[1][0]
    x2x2=A[1][1]

    B=np.subtract(np.array(b1),np.array(b2))

    x1=B[0]

    x2=B[1]
    
    c0=c1-c2

    print(x1x1,x1x2,x2x2,x1,x2,c0)

    x = np.arange(-1, 10, 0.001)

    posx=[]
    posy=[]
    negx=[]
    negy=[]

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
    plt.xlabel("x1")
    plt.ylabel("x2")
    #print(triangles)
    for i in range(0,6):
        print(w1[i][0],w1[i][1])
        c1=plt.scatter(w1[i][0], w1[i][1], color="green",marker='o')
    uc1=plt.scatter(mean1[0],mean1[1],color="black",marker='+')
    for i in range(0,6 ):
        print(w2[i][0],w2[i][1])
        c2=plt.scatter(w2[i][0], w2[i][1], color="red",marker='o')
    uc2=plt.scatter(mean2[0],mean2[1],color="blue",marker='+')

    #f= c0 + x2*Y +x1*i + x2x2*Y*Y + x1x2*i*Y + x1x1*i*i
    test1=plt.scatter(5,8,marker='*',color="purple")
    test2=plt.scatter(7,3,marker='*',color="yellow")
    getClass(5,8,x1x1,x1x2,x2x2,x1,x2,c0)
    getClass(7,3,x1x1,x1x2,x2x2,x1,x2,c0)

    plt.show()

if __name__ == '__main__':
	main()
