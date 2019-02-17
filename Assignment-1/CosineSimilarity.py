import math
import re
from collections import Counter

file1=open('doc1.txt','r')
file2=open('doc2.txt','r')

alpha=re.compile(r'\w+')

def ToVector(text):
	words=alpha.findall(text)
	return Counter(words)

def cosine_similarity(vector1,vector2):
	intersection=set(vector1.keys()) & set(vector2.keys())
	numerator=sum([vector1[x]*vector2[x] for x in intersection])

	sum1=sum([vector1[x]**2 for x in vector1.keys()])
	sum2=sum([vector2[x]**2 for x in vector2.keys()])
	denomator= math.sqrt(sum1)*math.sqrt(sum2)

	if not denomator:
		return 0
	else:
		return numerator/denomator
		
text1=file1.read()
text2=file2.read()

vector1=ToVector(text1)
vector2=ToVector(text2)

cosine=cosine_similarity(vector1,vector2)
print ('cosine similarity:-',cosine) 