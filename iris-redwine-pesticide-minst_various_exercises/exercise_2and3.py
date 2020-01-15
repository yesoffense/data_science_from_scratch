import numpy as np

redwine_train = np.loadtxt('redwine_training.txt')
redwine_test = np.loadtxt('redwine_testing.txt')

"""
1. fixed acidity
2. volatile acidity
3. citric acid
4. residual sugar
5. chlorides
6. free sulfur dioxide 
7. total sulfur dioxide 
8. density
9. pH
10. sulfates 
11. alcohol
"""


### Exercise 2

#a
def multivarlinreg(X, y):
    return np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(y,X))

#appending Xs and Ys
redwine_XTrain = np.array([e[:-1] for e in redwine_train])
redwine_YTrain = np.array([e[-1] for e in redwine_train])

vec1 = np.ones((redwine_YTrain.shape[0],1))
redwine_XTrain = np.concatenate((vec1, redwine_XTrain), axis = 1)

redwine_XTrain1 = np.array([e[:2] for e in redwine_XTrain])



# b
redwine_XTrain_frst_w = multivarlinreg(redwine_XTrain1,redwine_YTrain)
print('2b, weights for bias and first feature: ')
print('bias: ', redwine_XTrain_frst_w[0], 4)
print('fixed acidity: ', redwine_XTrain_frst_w[1], 4)

# c
redwine_XTrain_weights = multivarlinreg(redwine_XTrain, redwine_YTrain)
labels = ['bias: ', 'fixed acidity: ', 'volatile acidity: ', 'citric acid: ', 'residual sugar: ', 'chlorides: ', 'free sulfur dioxide: ', 'total sulfur dioxide: ', 'density: ', 'pH: ', 'sulfates: ', 'alcohol: ']
print('2c, weights for bias and all features:')

for idx in range(len(redwine_XTrain_weights)):
    print(labels[idx], redwine_XTrain_weights[idx], 4)


### EXERCISE 3

#a

def make_prediction(XTrain, YTrain, XTest):
    w = multivarlinreg(XTrain,YTrain)
    predicted = np.array([np.sum(e*w) for e in XTest])
    return predicted

def rmse(Y, Y_pred):
    return np.sqrt(sum((Y - Y_pred)**2)/len(Y)) 


redwine_XTest = np.array([e[:-1] for e in redwine_test])
redwine_YTest = np.array([e[-1] for e in redwine_test])

vec1 = np.ones((redwine_YTest.shape[0],1))
redwine_XTest = np.concatenate((vec1, redwine_XTest), axis = 1)

redwine_XTest1 = np.array([e[:2] for e in redwine_XTest])




#b
p = make_prediction(redwine_XTrain1,redwine_YTrain,redwine_XTest1)
print('3b, rmse for w1: ', rmse(p,redwine_YTest), 4)


#c
p = make_prediction(redwine_XTrain,redwine_YTrain,redwine_XTest)
print('3c, rmse for all features: ',rmse(p,redwine_YTest), 4)
