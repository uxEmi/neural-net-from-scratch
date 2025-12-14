import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

aux = pd.read_csv("diabetes_prediction_dataset.csv")
aux['gender'] = aux['gender'].map({'Female' : 0,'Male' : 1,'Other':2})
aux['smoking_history'] = aux['smoking_history'].map({'No Info' : 0,'never' : 1,'current' : 2,'former' : 3,'not current' : 4,'ever' : 5})

aux = aux.to_numpy()

'''
    numai am facut plot deoarece daca nu am corelatii puternice intre variabilele independente nu o sa arate frumos separat
'''

X = aux[:,:-1]
y = aux[:,-1]


X_train, X_aux, y_train, y_aux = train_test_split(X, y, train_size=0.8, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_aux, y_aux, train_size=0.5, random_state=42)

print(np.corrcoef(X_train, rowvar=False))

del X_aux
del y_aux

scaler = StandardScaler().fit(X_train,y_train)
X_train = scaler.transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)


model = LogisticRegression().fit(X_train,y_train)
y_train_pred = model.predict(X_train)
y_dev_pred = model.predict(X_dev)
y_test_pred = model.predict(X_test)

print(f'Train acc: {accuracy_score(y_train.flatten(),y_train_pred) * 100}')
print(f'Dev acc: {accuracy_score(y_dev.flatten(),y_dev_pred) * 100}')
print(f'Test acc: {accuracy_score(y_test.flatten(),y_test_pred) * 100}')
def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):
    learning_rate = learning_rate0 / (1 + decay_rate * (epoch_num // time_interval))
    return learning_rate

def init_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
    return v, s

def random_mini_b(X, Y, batch_size=256):
    m = X.shape[1]
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]
    mini_batches = []
    num_complete = m // batch_size
    for k in range(num_complete):
        mbX = X_shuffled[:, k*batch_size : (k+1)*batch_size]
        mbY = Y_shuffled[:, k*batch_size : (k+1)*batch_size]
        mini_batches.append((mbX, mbY))
    if m % batch_size != 0:
        mbX = X_shuffled[:, num_complete*batch_size :]
        mbY = Y_shuffled[:, num_complete*batch_size :]
        mini_batches.append((mbX, mbY))
    return mini_batches

def sigmoid(z):
    A = 1/(1 + np.exp(-z))
    A = np.clip(A, 1e-15, 1 - 1e-15)
    return A, z

def relu(z):
    A = np.maximum(0, z)
    return A, z

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
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(dims[l], dims[l-1]) * np.sqrt(2/dims[l-1])
        parameters['b' + str(l)] = np.zeros((dims[l], 1))
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def dropout_forward(A, keep_prob):
    D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
    A = A * D / keep_prob
    cache = (D, keep_prob)
    return A, cache

def dropout_backward(dA, cache):
    D, keep_prob = cache
    dA = dA * D / keep_prob
    return dA

def linear_activation_forward(A_prev, W, b, activation, keep_prob=1.0):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    if keep_prob < 1.0:
        A, dropout_cache = dropout_forward(A, keep_prob)
    else:
        dropout_cache = None
    cache = (linear_cache, activation_cache, dropout_cache)
    return A, cache

def L_model_forward(X, parameters, training=True, keep_prob=0.5):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        kp = keep_prob if training else 1.0
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu", kp)
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid", keep_prob=1.0)
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -1/m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL)))
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache, dropout_cache = cache
    if dropout_cache is not None:
        dA = dropout_backward(dA, dropout_cache)
    if activation == "relu":
        dZ = relu_backward(dA, (linear_cache, activation_cache))
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, (linear_cache, activation_cache))
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads['dA' + str(L-1)] = dA_prev_temp
    grads['dW' + str(L)] = dW_temp
    grads['db' + str(L)] = db_temp
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], current_cache, "relu")
        grads['dA' + str(l)] = dA_prev_temp
        grads['dW' + str(l + 1)] = dW_temp
        grads['db' + str(l + 1)] = db_temp
    return grads

def update_parameters(parameters, grads, learning_rate, v, s, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    parameters_copy = copy.deepcopy(parameters)
    for l in range(1, L + 1):
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1 ** t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1 ** t)
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (grads["dW" + str(l)] ** 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (grads["db" + str(l)] ** 2)
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2 ** t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2 ** t)
        parameters_copy["W" + str(l)] -= learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters_copy["b" + str(l)] -= learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)
    return parameters_copy, v, s

def predict(X, parameters):
    AL, _ = L_model_forward(X, parameters, training=False)
    return (AL > 0.5).astype(int).flatten()

X_train_updated = X_train.T
y_train_updated = y_train.reshape(1, -1)
X_dev_updated = X_dev.T
y_dev_updated = y_dev.reshape(1, -1)
X_test_updated = X_test.T
y_test_updated = y_test.reshape(1, -1)
dims = [X_train_updated.shape[0], 256, 128, 64, 1]

def L_layer_model(X, Y, dims, learning_rate=0.001, epoch=3000, print_cost=False, decay_rate=0.1, time_interval=1000, keep_prob=0.5):
    np.random.seed(1)
    parameters = initialize_parameters_deep(dims)
    v, s = init_adam(parameters)
    t = 1
    learning_rate0 = learning_rate
    for i in range(0, epoch):
        cost = 0
        mini_batches = random_mini_b(X, Y)
        current_lr = schedule_lr_decay(learning_rate0, i, decay_rate, time_interval)
        for minibatch in mini_batches:
            mbX, mbY = minibatch
            AL, caches = L_model_forward(mbX, parameters, training=True, keep_prob=keep_prob)
            cost += compute_cost(AL, mbY)
            grads = L_model_backward(AL, mbY, caches)
            parameters, v, s = update_parameters(parameters, grads, current_lr, v, s, t)
            t += 1
        if print_cost and i % 100 == 0:
            print(f"Epoch {i}, Cost: {cost / len(mini_batches):.4f}")
    return parameters


parameters = L_layer_model(X_train_updated, y_train_updated, dims, print_cost=True, keep_prob=0.7)
y_train_pred = predict(X_train_updated, parameters)
y_dev_pred = predict(X_dev_updated, parameters)
y_test_pred = predict(X_test_updated, parameters)
print(f'Train acc: {accuracy_score(y_train_updated.flatten(), y_train_pred) * 100:.2f}%')
print(f'Dev acc: {accuracy_score(y_dev_updated.flatten(), y_dev_pred) * 100:.2f}%')
print(f'Test acc: {accuracy_score(y_test_updated.flatten(), y_test_pred) * 100:.2f}%')
