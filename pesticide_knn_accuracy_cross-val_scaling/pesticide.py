import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from scipy.spatial.distance import euclidean

dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')

XTrain = dataTrain [:,:-1] 
#print(XTrain[0])
YTrain = dataTrain [:,-1]
XTest = dataTest[:,:-1] 
YTest = dataTest[:,-1]


### EXERCISE 1: 1-NN

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


# Function for accuracy score
  
def make_predictions(test_data, test_labels, training_data, training_labels, k):
  y_pred = []

  for dp_counter, datapoint in enumerate(test_data):
    y_pred.append(knn(datapoint, training_data, training_labels, k))
  
  accTest = accuracy_score(test_labels, y_pred)
  return accTest

print('accuracy for test set when k=1: ',(make_predictions(XTest, YTest, XTrain, YTrain, 1)))
print('accuracy for train set when k=1: ', (make_predictions(XTrain, YTrain, XTrain, YTrain, 1)))





### EXERCISE 2: CROSS VALIDATION
def cross_validation(trainset, trainlabels, k):
  cv = KFold(n_splits = 5)
  scores = []

  for train, test in cv.split(XTrain):
    XTrainCV, XTestCV, YTrainCV, YTestCV = trainset[train], trainset[test], trainlabels[train], trainlabels[test] 
    #print(XTestCV.shape) #80% = train, 20% = test.  
    temp_score = make_predictions(XTestCV, YTestCV, XTrainCV, YTrainCV, k)
    scores.append(temp_score)
  return sum(scores)/(len(scores))


k_s = [1,3,5,7,9,11]
k_scores = []

# Extracting values for each k
for k in k_s:
  #print('cross validation when k=', k, ' is: ', cross_validation(XTrain, YTrain, k))
  dict_k_best = {'k':k,'crossval': cross_validation(XTrain, YTrain, k)} 
  k_scores.append(dict_k_best)

# Deciding best number for k
best_k = {'k': 0, 'crossval': 0}
for scores in k_scores:
  if scores['crossval'] > best_k['crossval']:
    best_k = scores
print('best value for k:', best_k)






### EXERCISE 3

print('test accuracy when k=3: ',(make_predictions(XTest, YTest, XTrain, YTrain, 3)))
print('train accuracy when k=3: ', (make_predictions(XTrain, YTrain, XTrain, YTrain, 3)))




### EXERCISE 4

scaler = StandardScaler()
scaler.fit(XTrain)

XTrain = scaler.transform(XTrain)
#print(XTrain[0])
print(XTrain.mean(axis=0))
XTest = scaler.transform(XTest)  

print('test accuracy after normalization when k=3: ',(make_predictions(XTest, YTest, XTrain, YTrain, 3)))
print('train accuracy after normalization when k=3: ', (make_predictions(XTrain, YTrain, XTrain, YTrain, 3)))