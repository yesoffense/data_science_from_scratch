import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

### EXERCISE 9: k-Means
#28 × 28 = 784 pixels

MNIST_digits = np.loadtxt('MNIST_179_digits.txt')
MNIST_labels = np.loadtxt('MNIST_179_labels.txt')
#full_MNIST = np.concatenate((MNIST_labels[:,None], MNIST_digits), axis=1)

X_train, X_test, y_train, y_test = train_test_split(MNIST_digits , MNIST_labels, test_size=0.33, random_state=42)

kmeans = KMeans(n_clusters=3, random_state=0).fit(X_train )
cluster_centers = kmeans.cluster_centers_
pred_labels = kmeans.predict(X_test)

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

states = [dict(zip(labels0, counts0)), dict(zip(labels1, counts1)), dict(zip(labels2, counts2))]

for i in range(3):
    image = cluster_centers[i].reshape((28,28))

    plt.figure()
    plt.imshow(image, cmap='Greys')
    plt.title('Centroid {}.'.format(i))
    plt.savefig("centroid{}.png".format(i))

    print("cluster center {} has labels: ".format(i), states[i])

#b

"""
#own knn
def knn(input_value, training_data, traning_labels, k):
  distances = []

  for counter, features in enumerate(training_data): 
    each_dist = euclidean(input_value, features)
    #distances[counter,0] = each_dist
    #distances[counter,1] = traning_labels[counter]
    distances.append((each_dist,traning_labels[counter]))
  sorted_distances = sorted(distances, key=lambda value: value[0])

  top_k = sorted_distances[:k]
  top_k = np.asarray(top_k)
  mean_per_column = np.mean(top_k, axis=0)
  #print(mean_per_column)
  #print(mean_per_column[1])

  mean_class = mean_per_column[1]
  if mean_class <= 0.5:
    suggested_class = 0
  else:
    suggested_class = 1
  return suggested_class

#accuracy score
def make_predictions(test_data, test_labels, training_data, training_labels, k):
  y_pred = []

  for dp_counter, datapoint in enumerate(test_data):
    y_pred.append(knn(datapoint, training_data, training_labels, k))
  
  accTest = accuracy_score(test_labels, y_pred)
  return accTest

print('accuracy for test set when k=3: ',(make_predictions(XTest, YTest, XTrain, YTrain, 3)))
print('accuracy for train set when k=3: ', (make_predictions(XTrain, YTrain, XTrain, YTrain, 3)))
"""

X_train, X_test, y_train, y_test = train_test_split(MNIST_digits , MNIST_labels, test_size=0.5)
cv = KFold(n_splits = 5)

ks = [1, 3, 5, 7, 9, 11]
error = []
score = []
score_mean=[]

for k in ks:
    #cross-validation 
    for train, test in cv.split(X_train):
        XTrainCV, XTestCV, YTrainCV, YTestCV = X_train[train], X_test[test], y_train[train], y_test[test]
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(XTrainCV, YTrainCV) 
        k_predictions = knn.predict(XTestCV)
        # test accuracy of each k in each fold
        score.append(accuracy_score(YTestCV, k_predictions))

    #mean error and mean accuracy
    error.append((np.mean(k_predictions != YTestCV), k))
    score_mean.append((np.mean(score), k))
   
    
top_k = (sorted(error)[0])[1]

knn_k_best = KNeighborsClassifier(n_neighbors=top_k)
knn_k_best.fit(X_train, y_train) 
accTest = accuracy_score(y_test, knn_k_best.predict(X_test))
print("best k = " + str(top_k))
print("accuracy  = ", accTest)
