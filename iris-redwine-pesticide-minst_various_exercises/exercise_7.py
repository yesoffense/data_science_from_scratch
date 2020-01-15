import numpy as np
import matplotlib.pyplot as plt

train_2D1 = np.loadtxt('Iris2D1_train.txt')
test_2D1 = np.loadtxt('Iris2D1_test.txt') 

train_2D2 = np.loadtxt('Iris2D2_train.txt')
test_2D2 = np.loadtxt('Iris2D2_test.txt')

#For logistic regression, 7.b
x_train = train_2D2[:,:-1]
y_train = train_2D2[:,-1]  
y_train = y_train.reshape(-1,1)
 
x_test = test_2D2[:,:-1]
y_test = test_2D2[:,-1]
y_test = y_test.reshape(-1,1)


### EXERCISE 7

############# SCATTERPLOTS
#1

#2D1
#Concatenating datasets
iris_2D1 = np.concatenate((train_2D1, test_2D1), axis = 0)
labels = iris_2D1[:,-1]

#Scatterplot
plt.figure()
for idx in range(len(iris_2D1)): 
    if labels[idx] == 1: 
        plt.scatter(iris_2D1[idx, 0], iris_2D1[idx, 1], color='red')
    elif labels[idx] == 0.: 
        plt.scatter(iris_2D1[idx, 0], iris_2D1[idx, 1], color='green')
plt.title('2D1')
plt.xlabel('Sepal length and width in cm')
plt.ylabel('Petal length and width in cm')
plt.savefig('2D1_scatter.png')


#2D2  
#Concatenating datasets
iris_2D2 = np.concatenate((train_2D2, test_2D2), axis = 0)
labels = iris_2D2[:,-1]

#Scatterplot
plt.figure()
for idx in range(len(iris_2D2)): 
    if labels[idx] == 1: 
        plt.scatter(iris_2D2[idx, 0], iris_2D2[idx, 1], color='red')
    elif labels[idx] == 0.: 
        plt.scatter(iris_2D2[idx, 0], iris_2D2[idx, 1], color='green')
plt.title('2D2')
plt.xlabel('Sepal length and width in cm')
plt.ylabel('Petal length and width in cm')
plt.savefig('2D2_scatter.png')




############# IMPLEMENTING LOGISTIC REGRESSION 
#2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
 
def loss_func(y, h):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
            #or -np.mean(y*np.log(h)+(1-y)*np.log(1-h))
 
W = np.zeros((2,1))
b = np.zeros((1,1))
learning_rate = 0.01
 
m = len(y_train)
 
for epoch in range(5000):
    Z = np.matmul(x_train, W) + b
    A = sigmoid(Z)
    loss = loss_func(y_train, A)
    dz = A - y_train
    dw = 1/m * np.matmul(x_train.T, dz)
    db = np.sum(dz)
     
    W = W-learning_rate * dw
    b = b- learning_rate * db
     
 
def predictions(data_input, W, b):
    Z_train = np.matmul(data_input, W) + b
    predictions = []
    for i in sigmoid(Z_train):
        if i > 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions
 
def accuracy_func(predictions, labels):
    valid = 0
    invalid = 0
    for i in range((len(predictions))):
        if predictions[i] == labels[i]:
            valid += 1
        else:
            invalid += 1
    loss_function = valid/len(predictions)
    return loss_function
 


############### RESULTS
#3-4

#predictions and loss for trainset and data in scatterplot
train_predictions = predictions(x_train, W, b)
print('predictions, training data:', train_predictions)
 
print('accuracy, training data', accuracy_func(train_predictions, y_train))

plt.figure() 
plt.scatter(x_train[:,0], x_train[:,1], c = y_train.ravel())
ax = plt.gca()
x_values = np.array(ax.get_xlim()).reshape(-1,1)
y_values = -(x_values * W[0][0] + b)/ W[1][0]
plt.title('Predictions and loss for train set')
plt.plot(x_values, y_values)
plt.savefig('train_preds_and_loss.png')

print('bias for train set: ', b)
print('weights for train set: ', W)

#predictions and loss for testset and data in scatterplot
test_predictions = predictions(x_test, W, b)
print('predictions, test data: ', test_predictions)
print('accuracy, test data', accuracy_func(test_predictions, y_test))


plt.figure() 
plt.scatter(x_test[:,0], x_test[:,1], c = y_test.ravel())
ax = plt.gca()
x_values = np.array(ax.get_xlim()).reshape(-1,1)
y_values = -(x_values * W[0][0] + b)/ W[1][0]
plt.plot(x_values, y_values)
plt.title('Predictions and loss for test set')
plt.savefig('test_preds_and_loss.png') 
 
print('bias for test set: ', b)
print('weights for test set: ', W)
