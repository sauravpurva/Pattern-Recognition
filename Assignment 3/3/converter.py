#Assignment-3.3
#Group-1
from __future__ import division
from PIL import Image
import numpy as np
import pickle

def getColorAndSize(img):
    colors = img.convert("RGB").getcolors(img.size[0]*img.size[1])#Set the maxcolors number to the image's pixels number.
    sum=0
    total=0
    eps = 1.5
    colour=0
    # print(colors)

    for x in colors:
        if (np.mean(x[1])<210):
            # print(colors)
            sum+=x[0]
            if( x[1][0] > eps*np.mean(x[1]) ):
                colour = colour + 1*x[0]
            if( x[1][1] > eps*np.mean(x[1]) ):
                colour = colour + 2*x[0]
        total+=x[0]
    colour = colour / sum
    #print(sum,total)
    percentage=(sum/total)
    return (colour,percentage)


def main():
    triangles=[]
    circles=[]
    for i in range(1,11):
        img1=Image.open('im'+str(i)+'.jpg')
        img2=Image.open('img'+str(i)+'.jpg')
        triangles.append(getColorAndSize(img1))
        circles.append(getColorAndSize(img2))
    print("Triangles:\n",triangles,"\n\nCircles:\n",circles)

    with open('triangles.pkl', 'wb') as f:
        pickle.dump(triangles, f,2)

    with open('circles.pkl', 'wb') as f:
        pickle.dump(circles, f,2)


if __name__ == '__main__':
	main()
