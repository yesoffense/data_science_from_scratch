import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import cov
from numpy.linalg import eig
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans

pesticide = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
pesticide = pesticide[:,:-1] 
murderdata = np.loadtxt('murderdata2d.txt')

### EXERCISE 1: PCA

#a: PCA function

def pca(datamatrix):
  #Centering = the average value for each variable will be 0
  mean_per_column = np.mean(datamatrix, axis=0) 
  centered_data = (datamatrix - mean_per_column)
  datamatrix = centered_data.T

  Sigma = np.cov(datamatrix)
  eigenValues, eigenVectors = np.linalg.eig(Sigma)

  #Sorting
  idx = np.argsort(eigenValues)[::-1]   
  eigenValues = eigenValues[idx]
  eigenVectors = eigenVectors[:,idx]
  
  return eigenValues,eigenVectors,centered_data


### b: Scatterplot

murder_eigenValues, murder_eigenVectors, murder_normdata = pca(murderdata)
print('eigenValues and eigenValues for murderset: ', murder_eigenValues, murder_eigenVectors)

x,y = zip(*murder_normdata)
plt.figure()

plt.scatter(x,y)
plt.axis('equal')

s0 = np.sqrt(murder_eigenValues[0])
s1 = np.sqrt(murder_eigenValues[1])

plt.plot([0, s0*murder_eigenVectors[0,0]], [0, s0*murder_eigenVectors[1,0]], 'r')
plt.plot([0, s1*murder_eigenVectors[0,1]], [0, s1*murder_eigenVectors[1,1]], 'r')

plt.xlabel('unemployed')
plt.ylabel('murder')
plt.title('PCA on murderset')
plt.savefig('pca_murder_scatter.png')



### c: PCA on the pesticide dataset

pesticide_eigenValues, pesticide_eigenVectors, pesticide_normalizedDatamatrix = pca(pesticide)

# variance
plt.figure()
plt.plot(pesticide_eigenValues)
plt.xlabel('principal components in descending order')
plt.ylabel('projected variance')
plt.title('variance versus principal components')
plt.savefig('pesticide_variance.png')

# cumulative variance
cumulative_variance = np.cumsum(pesticide_eigenValues/np.sum(pesticide_eigenValues))

plt.figure()
plt.plot(cumulative_variance)
plt.xlabel('principal components')
plt.ylabel('projected variance')
plt.title('cumulative variance versus principal components')
plt.savefig('pesticide_cumulative_variance.png')





### EXERCISE 2: MDS

def mds(data, dimensions): #1) datamatrix and 2) an integer d specifying the number of dimensions for the output (most commonly used are 2 or 3)
    eigenValues, eigenVectors, scaledData = pca(data)

    best_preserved = eigenVectors[:,:dimensions].T
    cords = best_preserved.dot(scaledData.T).T

    plt.figure()
    plt.scatter(cords[:,0],cords[:,1])
    plt.xlabel("first principal component")
    plt.ylabel("second principal component")
    plt.axis('equal')
    plt.savefig('mds_pesticide.png')

    # output:  1) an N x d numpy array containing the d coordinates of the N original datapoints projected onto the top d PCs
    return cords

print(mds(pesticide, 2))





### EXERCISE 3: K-MEANS

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

  print('centroid 1: ', centroid1)
  print('centroid 2: ', centroid2)

  return centroid1, centroid2

k_means(pesticide, 70)

#k-Means using scikitlearn for comparison

startingPoint = np.vstack((pesticide[0,] ,pesticide[1,]))
kmeans = KMeans(algorithm='full',n_init=1, init=startingPoint, max_iter=70, n_clusters=2, tol=0.0001).fit(pesticide)
print(kmeans.cluster_centers_)
