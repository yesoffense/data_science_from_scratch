from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


MNIST_digits = np.loadtxt("MNIST_179_digits.txt")
MNIST_labels = np.loadtxt("MNIST_179_labels.txt")
#Making training and test set:
digits_XTrain = MNIST_digits[:900]
digits_XTest = MNIST_digits[900:]
labels_YTrain = MNIST_labels[:900]
labels_YTest = MNIST_labels[900:]


### EXERCISE 10
#a

pca = PCA().fit(digits_XTrain)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_), label="Captured variance by PCs")

plt.xlabel("PCs sorted by decreasing variance")
plt.ylabel("Cumulative percentage of variance")
plt.xticks((0, 199, 399, 599, 783), (1, 200, 400, 600, 784))

plt.savefig("10a.png")

#10b
"""
def pca(data):
    #centering
    mean_data = np.mean(data, axis = 0)
    centered = data-mean_data
    Sigma = np.cov(centered.T)
    evals, evecs = np.linalg.eig(Sigma)
    
    #sorting the values
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
    return evals, evecs

def mds(data, dimensions):
    evals, evecs = pca(data)
    #preparing the top dimensions for the following dot-product calculation
    top_dimensions = evecs[:, :dimensions]
    #calculating the dot-product of the dimension vectors and the data
    new_coordinates = np.dot(data, top_dimensions)
    # output:  1) an N x d numpy array containing the d coordinates of the N original datapoints projected onto the top d PCs
    return new_coordinates, top_dimensions


projected_digits_XTrain, digits_top_dimensions = mds(digits_XTrain, 2)
projected_both_XTrains = np.concatenate((labels_YTrain[:,None], projected_digits_XTrain), axis=1)




kmeans = KMeans(n_clusters=3, random_state=0).fit(projected_both_XTrains)
cluster_centers = kmeans.cluster_centers_
pred_labels = kmeans.predict(projected_both_XTrains)

zero = []
one = []
two = []

for i, element in enumerate(pred_labels):
    if element == 0:
        zero.append(i)
    elif element == 1:
        one.append(i)
    else:
        two.append(i)
        
        
cluster_0 = y_test[zero]
cluster_1 = y_test[one]
cluster_2 = y_test[two]

labels0, counts0 = np.unique(cluster_0, return_counts=True)
labels1, counts1 = np.unique(cluster_1, return_counts=True)
labels2, counts2 = np.unique(cluster_2, return_counts=True)



centroids = np.vstack((centroid_0, centroid_1, centroid_2))
reconstructed_centroids = np.dot(centroids, digits_top_dimensions.T).real.astype(float)
"""


"""
#c
knn_k_best = KNeighborsClassifier(n_neighbors=top_k)
best_k = knn_k_best.fit(digits_XTrain, labels_YTrain) 
print("best k = " + str(top_k))

print("best k is", best_k)
"""