import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

IMG_SIZE = 64  

ds = tfds.load("cats_vs_dogs", as_supervised=True)["train"]

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    return image, label

ds = ds.map(preprocess)

X = np.array([img for img, lbl in tfds.as_numpy(ds)])
y = np.array([lbl for img, lbl in tfds.as_numpy(ds)])

X = X.reshape(X.shape[0],-1)
y = y.ravel()

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

X_train, X_aux, y_train, y_aux = train_test_split(X_scaled, y, train_size=0.6, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_aux, y_aux, train_size=0.5, random_state=42)

del X_aux
del y_aux

model_rl = LogisticRegression(random_state=0).fit(X_train,y_train)
y_pred_train = model_rl.predict(X_train)
y_pred_dev = model_rl.predict(X_dev)
y_pred_test = model_rl.predict(X_test)

print(f'Rezultate pentru regresia logistica:')
print(f'Train acc: {accuracy_score(y_train,y_pred_train) * 100}')
print(f'Dev acc: {accuracy_score(y_dev,y_pred_dev) * 100}')
print(f'Test acc: {accuracy_score(y_test,y_pred_test) * 100}')

def predict(X,parameters):
    AL, _ = L_model_forward(X,parameters) 
    return (AL > 0.5).astype(int).flatten()

def sigmoid(z):
    A =  1/(1 + np.exp(-z))
    return A,z

def relu(z):
    A = np.maximum(0,z)
    return A,z

def sigmoid_backward(dA, cache):
    linear_cache, activation_cache = cache
    Z = activation_cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def relu_backward(dA, cache):
    linear_cache, activation_cache = cache
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def initialize_parameters_deep(dims):
    np.random.seed(1)
    parameters = {}
    L = len(dims)
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(dims[l],dims[l-1]) * np.sqrt(2/dims[l-1])
        parameters['b' + str(l)] = np.zeros((dims[l],1))
    return parameters

def linear_forward(A,W,b):
    Z = np.dot(W,A) + b 
    cache = (A,W,b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,parameters['W' + str(l)],parameters['b' + str(l)],"relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A,parameters['W' + str(L)],parameters['b' + str(L)],"sigmoid")
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -1/m * np.sum(np.multiply(Y,np.log(AL)) + np.multiply((1-Y),np.log(1-AL)))
    cost = np.squeeze(cost)

    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1/m * np.dot(dZ,A_prev.T)
    db = 1/m * np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, cache)       
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, cache)        
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL,current_cache,"sigmoid")
    grads['dA' + str(L-1)] = dA_prev_temp
    grads['dW' + str(L)] = dW_temp
    grads['db' + str(L)] = db_temp
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)],current_cache,"relu")
        grads['dA' + str(l)] = dA_prev_temp
        grads['dW' + str(l + 1)] = dW_temp
        grads['db' + str(l + 1)] = db_temp
    
    return grads

def update_parameters(params, grads, learning_rate):
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
    
    return parameters

X_train_updated = X_train.T
y_train_updated = y_train.reshape(1,-1)
X_dev_updated = X_dev.T
y_dev_updated = y_dev.reshape(1,-1)
X_test_updated = X_test.T
y_test_updated = y_test.reshape(1,-1)

dims = [X_train_updated.shape[0],15,10,1]
def L_layer_model(X, Y, dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X,parameters)
        cost = compute_cost(AL,Y)
        grads = L_model_backward(AL,Y,caches)
        parameters = update_parameters(parameters,grads,learning_rate)
        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0:
            costs.append(cost)

    return parameters, costs

parameters, costs = L_layer_model(X_train_updated, y_train_updated, dims, num_iterations = 1000, print_cost = True)

y_train_pred2 = predict(X_train_updated,parameters)
y_dev_pred2 = predict(X_dev_updated,parameters)
y_test_pred2 = predict(X_test_updated,parameters)

print("Rezultate cu retea neuronala mare :")
print(f'Train acc: {accuracy_score(y_train_updated.flatten(),y_train_pred2) * 100}')
print(f'Dev acc: {accuracy_score(y_dev_updated.flatten(),y_dev_pred2) * 100}')
print(f'Test acc: {accuracy_score(y_test_updated.flatten(),y_test_pred2) * 100}')


