import numpy as np
import matplotlib.pyplot as plt

ids_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
ids_test = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')

XTrain = ids_train[:,:-1]
YTrain = ids_train[:,-1] 
XTest = ids_test[:, :-1]
YTest = ids_test[:, -1]

def f(x):
    return (np.exp(-x/2)+10*(x**2))

def derivative(x):
    return (20*x-0.5*np.exp(-x/2))

def gradient_descent():
    learning_rates = [0.1, 0.01, 0.001, 0.0001]

    for rate in learning_rates:
        x = 1
        for i in range(10000): # Number of iterations
            dx = derivative(x)
            x-=dx*rate
            if abs(dx)<10**-10: # Tolerance set to 10e**-10
                print('rate: {}:\nnumber of iterations it took the algorithm to converge: {}\nfunction value at convergence: {}\n'.format(rate, i, f(x)))
                break
            elif i == 9999:
                print('rate:{} \nno more iterations'.format(rate))


def tangent(a, x): 
    return f(a)+derivative(a)*(x-a)

def tangent_plot():
    fig, axes = plt.subplots(4)
    fig.set_size_inches(5, 15)
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    cm = plt.get_cmap('gist_rainbow')
    for idx,rate in enumerate(learning_rates):
        lims = (-1, 1)
        xs = np.linspace(*lims, 25)
        x0 = 1
        for i in range(3):
            ys = [tangent(x0, x) for x in xs]
            x0-=derivative(x0)*rate
            ax=axes[idx]
            ax.plot(xs, ys, color=cm(i*20))
            ax.plot(xs,f(xs),color='black')
            ax.scatter(x0,f(x0),marker='o',s=100,color=cm(i*20))
            ax.set_title('tangent lines and steps for learning rate: {}'.format(rate))
    plt.savefig('tangets.png')


def steps_plot():
    fig, axes = plt.subplots(4)
    fig.set_size_inches(5, 15)
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    cm = plt.get_cmap('gist_rainbow')
    for idx,rate in enumerate(learning_rates):
        lims = (-1, 1)
        xs = np.linspace(*lims, 25)
        x0 = 1
        ax=axes[idx]
        for i in range(10):
            x0-=derivative(x0)*rate
            ax.plot(xs,f(xs),color='black')
            ax.scatter(x0,f(x0),marker='o',s=100,color=cm(i*20))
            ax.set_title('10 steps for learning rate: {}'.format(rate))
    plt.savefig('steps.png')

gradient_descent()
tangent_plot()
steps_plot()
