
import numpy as np 
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt 
from scipy.signal import argrelextrema
import matplotlib.cm as cm 
import os 

def kde(data,bdw):
	'''Takes 1d data and performs unsupervised clustering using its densities.
		The number of clusters are determined by the bandwidth
		(variance of the gaussians)
		'''
	a = np.array(data).reshape(-1, 1)
	kde = KernelDensity(kernel='gaussian', bandwidth=bdw).fit(a)
	
	Highest = max(data)+10
	s = np.linspace(0,Highest)
	e = kde.score_samples(s.reshape(-1,1))
	mi,ma = argrelextrema(e,np.less)[0],(e,np.greater)[0]
	colors = cm.rainbow(np.linspace(0, 1, len(mi)+1))
	terms = []
	if len(mi)==0:
		return []

	'''PLOTTING'''
	for i in range(len(mi)+1):
		if i == 0:
			terms.append([s[:mi[i]+1],e[:mi[i]+1]])
		elif i>0 and i < len(mi):
			terms.append( [s[mi[i-1]:mi[i]+1], e[mi[i-1]:mi[i]+1]])
		elif i==len(mi):
			terms.append( [s[mi[i-1]+1:],	e[mi[i-1]+1:]])
	terms = np.array(terms)
	for val,colour in zip(terms,colors):
		plt.plot(val[0],val[1],color=colour)
	plt.title('1D Clustering of the X pixels using KDE')
	plt.xlabel('X-axis pixels')
	plt.ylabel('Densities')
	# plt.show()

	'''GROUPING THE DATA'''
	clusters = []
	for i in range(len(mi)+1):
		if i ==0:
			clusters.append( a[a<int(np.ceil(max (s[:mi[i]+1]) ) ) ]
							)
		elif i>0 and i<len(mi):
			
			clusters.append(a[ (a>=int(np.ceil(max(s[:mi[i-1]+1])))) * (a<int(np.ceil(max(s[:mi[i]+1])))) ] )
		elif i == len(mi):
			clusters.append(a[a>int(np.ceil(max(s[:mi[i-1]+1])))])

	return (clusters)


if __name__ == '__main__':
	data_1 = [10,11,9,23,21,11,45,20,11,12,46,47,96,97,98] #Actual - 4 
	data_2 = [74.5, 119.5, 70.5, 189.5, 180.5, 86.5, 70.5, 182.5, 180.5, 256.5, 45.5, 240.5, 251.5] #Actual - 7
	data_3 = [1,2,3,4,5,6] #Actual -6
	data_4 = [11.2,11.4,11.0,13.5,13.2,13.8,19.2,19.1,20.0,25.2] #Actual -4
	data_5 = [1,2,3,10,11,12,20,22,21] #Actual- 3

	#Testing 
	#print(kde(data_1,0.3))
	#print(kde(data_1,0.6))
	#print(kde(data_1,0.8))
	#print(kde(data_1,1))
	#print(kde(data_1,1.3))
	#print(kde(data_1,1.6))
	#print(kde(data_1,2))
	#print(kde(data_1,2.3))
	#print(kde(data_1,2.7))
	#print(kde(data_1,3))
	#print(kde(data_1,3.5))
	#print(kde(data_1,4))