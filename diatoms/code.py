from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import cov
from numpy.linalg import eig
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
import random
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

pesticide = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
labels = pesticide[:,-1] #last column

pesticide = pesticide[:,:-1] 

diatoms = np.loadtxt('diatoms.txt')
toy = np.loadtxt('pca_toydata.txt')


### EXERCISE 1
x = diatoms[0,0::2] # [first:start from 0(x):every other x]
y = diatoms[0,1::2]

plt.figure()
plt.plot(x,y) #interpolate instead of scatterplots
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cell Shape Of One Diatom')
plt.savefig('diatoms_plot.png')

plt.figure()

for i in range(779):
  i = i+1
  x_all = diatoms[i,0::2]
  y_all = diatoms[i,1::2]

    
  plt.plot(x_all,y_all) #interpolate instead of scatterplots
  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title('Cell Shape of All Diatoms')
  plt.savefig('all_diatoms_plot.png')



### EXERCISE 2

def pca(datamatrix):
  mean_per_column = np.mean(datamatrix, axis=0) 
  centered_data = (datamatrix - mean_per_column) #Centering = the average value for each variable will be 0
  datamatrix = centered_data.T

  Sigma = np.cov(datamatrix)
  eigenValues, eigenVectors = np.linalg.eig(Sigma)

  #Sorting
  idx = np.argsort(eigenValues)[::-1]   
  eigenValues = eigenValues[idx]
  eigenVectors = eigenVectors[:,idx]
  
  return eigenValues,eigenVectors,centered_data


variance, principal_components, centered_data = pca(diatoms)


plt.figure()

diatoms_mean = np.mean(diatoms, axis=0)

stddev1 = math.sqrt(variance[0])
stddev2 = math.sqrt(variance[1])
stddev3 = math.sqrt(variance[2])

#principal components = in columns
evecs1 = principal_components[:,0]
evecs2 = principal_components[:,1]
evecs3 = principal_components[:,2]


#cell 1
cell1 = diatoms_mean - 2 * stddev1*evecs1
cell2 = diatoms_mean - stddev1*evecs1
cell3 = diatoms_mean
cell4 = diatoms_mean + stddev1*evecs1
cell5 = diatoms_mean + 2 * stddev1*evecs1
cell1_x = cell1[0::2] #[first,start from 0(x),take every other x]
cell1_y = cell1[1::2]
cell2_x = cell2[0::2] 
cell2_y = cell2[1::2]
cell3_x = cell3[0::2] 
cell3_y = cell3[1::2]
cell4_x = cell4[0::2] 
cell4_y = cell4[1::2]
cell5_x = cell5[0::2] 
cell5_y = cell5[1::2]

plt.plot(cell1_x,cell1_y) #interpolate instead of scatterplots
plt.plot(cell2_x,cell2_y) 
plt.plot(cell3_x,cell3_y) 
plt.plot(cell4_x,cell4_y) 
plt.plot(cell5_x,cell5_y) 
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Spatial Variance of PC1')
plt.savefig('exercise2_plot1.png')

#cell 2
plt.figure()

cell1 = diatoms_mean - 2 * stddev2*evecs2
cell2 = diatoms_mean - stddev2*evecs2
cell3 = diatoms_mean
cell4 = diatoms_mean + stddev2*evecs2
cell5 = diatoms_mean + 2 * stddev2*evecs2
cell1_x = cell1[0::2] #[first,start from 0(x),take every other x]
cell1_y = cell1[1::2]
cell2_x = cell2[0::2] 
cell2_y = cell2[1::2]
cell3_x = cell3[0::2] 
cell3_y = cell3[1::2]
cell4_x = cell4[0::2] 
cell4_y = cell4[1::2]
cell5_x = cell5[0::2] 
cell5_y = cell5[1::2]

plt.plot(cell1_x,cell1_y) 
plt.plot(cell2_x,cell2_y) 
plt.plot(cell3_x,cell3_y) 
plt.plot(cell4_x,cell4_y) 
plt.plot(cell5_x,cell5_y) 
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Spatial Variance of PC2')
plt.savefig('exercise2_plot2.png')

#cell 3
plt.figure()

cell1 = diatoms_mean - 2 * stddev3*evecs3
cell2 = diatoms_mean - stddev3*evecs3
cell3 = diatoms_mean
cell4 = diatoms_mean + stddev3*evecs3
cell5 = diatoms_mean + 2 * stddev3*evecs3
cell1_x = cell1[0::2] 
cell1_y = cell1[1::2]
cell2_x = cell2[0::2] 
cell2_y = cell2[1::2]
cell3_x = cell3[0::2] 
cell3_y = cell3[1::2]
cell4_x = cell4[0::2] 
cell4_y = cell4[1::2]
cell5_x = cell5[0::2] 
cell5_y = cell5[1::2]

plt.plot(cell1_x,cell1_y) 
plt.plot(cell2_x,cell2_y) 
plt.plot(cell3_x,cell3_y) 
plt.plot(cell4_x,cell4_y) 
plt.plot(cell5_x,cell5_y) 
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Spatial Variance of PC3')
plt.savefig('exercise2_plot3.png')



### EXERCISE 3, b

def mds(data, dimensions): #1) datamatrix and 2) an integer d specifying the number of dimensions for the output (most commonly used are 2 or 3)
    eigenValues, eigenVectors, scaledData = pca(data)

    best_preserved = eigenVectors[:,:dimensions].T
    cords = best_preserved.dot(scaledData.T).T    
    # output:  1) an N x d numpy array containing the d coordinates of the N original datapoints projected onto the top d PCs
    return cords

toy_coords = mds(toy, 2)

plt.figure()
plt.axis('equal')
plt.scatter(toy_coords[:,0],toy_coords[:,1], s=2)
plt.xlabel("first principal component")
plt.ylabel("second principal component")
plt.title("All Data Points of Toy-dataset")
plt.savefig('toy.png')

plt.figure()
plt.axis('equal')
plt.scatter(toy_coords[:100,0],toy_coords[:100,1], s=2)
plt.xlabel("first principal component")
plt.ylabel("second principal component")
plt.title("Toy-dataset Without Last Two Data Points")
plt.savefig('toy_without_last_2.png')



### EXERCISE 4


def mds(data, dimensions): #1) datamatrix and 2) an integer d specifying the number of dimensions for the output (most commonly used are 2 or 3)
    eigenValues, eigenVectors, scaledData = pca(data)

    best_preserved = eigenVectors[:,:dimensions].T
    cords = best_preserved.dot(scaledData.T).T
    #cords_centroid1 = best_preserved.dot(centroid.T).T

    
    #plt.figure()
    #plt.scatter(cords[:,0],cords[:,1], color='red')
    #plt.xlabel("first principal component")
    #plt.ylabel("second principal component")
    #plt.axis('equal')
    #plt.savefig('mds_pesticide.png')
    
    # output:  1) an N x d numpy array containing the d coordinates of the N original datapoints projected onto the top d PCs
    return cords, best_preserved


#2-Means
def k_means(training_data, iterations):
  centroid1 = training_data[0]
  centroid2 = training_data[1]

  for iteration in range(iterations):
    cluster1 = []
    cluster2 = []

    for dp in training_data:
      if euclidean(dp, centroid1) > euclidean(dp, centroid2):
        cluster1.append(dp)
      else:
        cluster2.append(dp)

    #np.arrays
    cluster1 = np.asarray(cluster1)
    cluster2 = np.asarray(cluster2)

    #moving centroid to calculated mean
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)

  return centroid1, centroid2, cluster1, cluster2


eigenValues, principal_components, scaledData = pca(pesticide)
pesticide_coords, best_preserved = mds(pesticide, 2)
centroid1, centroid2, cluster1, cluster2 = k_means(pesticide, 30)


projected_centroid1 = np.dot(centroid1, principal_components)
print('projected centroid 1: ', projected_centroid1)
projected_centroid2 = np.dot(centroid2, principal_components)
print('projected centroid 2: ', projected_centroid2)


plt.figure()
plt.axis('equal')
idx_weed = [idx for idx in range(len(labels)) if labels[idx]==0] #indexes for weed
idx_crops = [idx for idx in range(len(labels)) if labels[idx]==1] #indexes for crop

# visualising clusters
plt.scatter(pesticide_coords[idx_weed,0], pesticide_coords[idx_weed,1], color='green')
plt.scatter(pesticide_coords[idx_crops,0], pesticide_coords[idx_crops,1], color='blue')
#visualising centroids
plt.scatter(projected_centroid1[0], projected_centroid1[1], c='red', s=100, label = "cluster centers")
plt.scatter(projected_centroid2[0], projected_centroid2[1], c='red', s=100, label = "cluster centers")

plt.title('Clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('vis_variance_own.png')












