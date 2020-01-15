import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as rnd
from scipy.spatial.distance import euclidean

dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')

XTrain = dataTrain [:,:-1] 
YTrain = dataTrain [:,-1]
XTest = dataTest[:,:-1] 
YTest = dataTest[:,-1]

"""
def kNN_accuracy(k):
	clf = KNeighborsClassifier(n_neighbors=k)
	clf.fit(x_train, y_train) 
	accuracy_score_test = accuracy_score(y_test, clf.predict(x_test))
	accuracy_score_train = accuracy_score(y_train, clf.predict(x_train))
	return accuracy_score_test,accuracy_score_train

"""

def knn(input_value, training_data, traning_labels, k):
  distances = []

  for counter, features in enumerate(training_data): 
    each_dist = euclidean(input_value, features)
    distances.append((each_dist,traning_labels[counter]))
  sorted_distances = sorted(distances, key=lambda value: value[0])

  top_k = sorted_distances[:k]
  top_k = np.asarray(top_k)
  mean_per_column = np.mean(top_k, axis=0)

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


def random_forest():
    clf = rnd(n_estimators=50,random_state=42)
    clf.fit(XTrain, YTrain)
    accuracy_score_test = accuracy_score(YTest, clf.predict(XTest))
    accuracy_score_train = accuracy_score(YTrain, clf.predict(XTrain))
    return accuracy_score_test,accuracy_score_train

# Exercise 3 and 4:
print('random forest accuracy: ',random_forest()[0], 3)
#print('k-NN accuracy: ',kNN_accuracy(1)[0], 4)