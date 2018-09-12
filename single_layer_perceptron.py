import numpy as np
import sklearn.datasets as datasets

from matplotlib import pyplot as plt
import seaborn as sns


'''
Goal: Use sklearn's breast cancer dataset to classify cell features as malignant or benign with a single layer perceptron NN.

To train the neuron:

1. Forward pass: Ask the neuron to classify a sample.
2. Update the neuron’s weights based on how wrong the prediction is. 
3. Repeat.
'''

# Functions 

def sigmoid(x):
    """
    Activation function for the single layer perceptron.
    Chosen because it is easily differentiable, non-linear, and bounded.
    """
    return 1/(1+np.exp(-x))


def dsigmoid(y):
    """
    Derivative of the sigmoid function. Useful for minimizing the error of the
    neuron's predictions.
    """
    return y*(1-y)


def mean_squared_error(y, pred):
    """
    Error function representing the scaled version of mean squared error.
    Function used to
    """
    return np.mean(np.square(y - pred))/2

# Step 1A: Load the data
breast_cancer_data = datasets.load_breast_cancer()

print('Class Labels: {}'.format(breast_cancer_data.target_names))
print('Dataset Shape: {}'.format(breast_cancer_data.data.shape))  # (569, 30) - (num_instances, num_features)
# print('Dataset features: {}'.format(breast_cancer_data.feature_names))
print('{} vals in sample: {}'.format(breast_cancer_data.target_names[0], np.count_nonzero(breast_cancer_data['target'] == 0)))
print('{} vals in sample: {}'.format(breast_cancer_data.target_names[1], np.count_nonzero(breast_cancer_data['target'] == 1)))
x = breast_cancer_data['data']  # array of features with shape: (n_samples, n_features)
y = breast_cancer_data['target']  # array of binary values for the two classes: (n_samples)

# Step 1B: Randomize the data
shuffle_ind = np.random.permutation(len(x))  # shuffle data for training

x = np.array(x[shuffle_ind])
y = np.array(y[shuffle_ind])[np.newaxis].T  # convert y to a column to make it more consistent with x

x /= np.max(x)  # linearly scaling the inputs to a range between 0 and 1, sigmoid function is bounded (0 to 1)

# Step 1C: Break the data into training, testing sets
train_fraction = 0.75
train_idx = int(len(x)*train_fraction)

x_train = x[:train_idx]
y_train = y[:train_idx]

x_test = x[train_idx:]
y_test = y[train_idx:]

# Step 1D: Initialize a single layer (vector) of weights with Python's matrix package numpy
W = 2*np.random.random((np.array(x).shape[1], 1)) - 1  # shape: (n_features * 1)


# Step 2B: Update the neuron's weights

# With the bounded predictions l1, need to calculate
# the error/loss in these predictions and update neuron weights
# proposal to this error
# We'll use the gradient descent optimization algorith for weight update
# To use it, need an error function to represent the gap b/w neuron's predictions and ground truth
# Error function is defined to be a scaled version of mean squared error
# A gradient is the derivative vector of a function that'd dependent on 1+ variables
# Vectors have a magnitude and a direction, the neuron's error depends on all weights feeding into it
# Gradient is the set of partial derivaties of error w/respect to each weight
# Moving the opposite direction of the gradient (the rate of most increase) minimizes the error
# therefore need to calculate the gradient of the error function for our neuron relative to each weight
# then update the weights proportionally 

lr = 0.5  # learning rate
history = {"loss": [], "val_loss": [], "acc": [],
           "val_acc": []}  # Metrics to track while training
epochs = 1000

for iter in range(epochs):  # Step 2C: Repeat!
    
    # Step 2A: Training - Forward Pass, ask the neuron to predict on the training samples
    l0 = x_train  # layer 0 output, the matrix of features shape:(n_samples * n_features)
    l1 = sigmoid(np.matmul(l0, W))  # perceptron output, or the class predictions for each training sample

    l1_error = l1 - y_train  # backward pass, output layer error

    l1_gradient = dsigmoid(l1)  # minimize the error w/derivative activation func
    
    l1_delta = np.multiply(l1_error, l1_gradient)  # calculate the weight update

    l1_weight_delta = np.matmul(l0.T, l1_delta)  # calculate the weight update

    W -= l1_weight_delta * lr  # update weights with a scaling factor of learning rate

    if iter % 100 == 0:
        # Recording metrics to understand the performance of neuron at each epoch
        train_loss = mean_squared_error(y_train, l1)
        train_acc = np.sum(y_train == np.around(l1))/len(x_train)
        history["loss"].append(train_loss)
        history["acc"].append(train_acc)

        val_pred = sigmoid(np.dot(x_test, W))
        val_loss = mean_squared_error(y_test, val_pred)
        val_acc = np.sum(y_test == np.around(val_pred))/len(x_test)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print('Epoch: {}, Training Loss: {}, Validation Acc: {}'.format(
            iter, train_loss, val_acc))

val_pred = sigmoid(np.dot(x_test, W))
print("Validation loss: {}".format(mean_squared_error(y_test, val_pred)))
print("Validation accuracy: {}".format(
    np.sum(y_test == np.around(val_pred))/len(x_test)))

# Plotting
sns.set()
fig = plt.figure()
fig.tight_layout()
plt_1 = plt.subplot(2, 1, 1)
plt.plot(history['loss'], label="Training Loss")
plt_1.set_ylabel('Loss', fontsize=10)
plt_1.set_title('Neuron training results', fontsize=10)
plt.plot(history['val_loss'], label="Validation Loss")
plt.legend()

plt_2 = plt.subplot(2, 1, 2)
plt.plot(history['acc'], label="Training Accuracy")
plt.plot(history['val_acc'], label="Validation Accuracy")
plt_2.set_xlabel('Epoch (100s)', fontsize=10)
plt_2.set_ylabel('Accuracy (%)', fontsize=10)
plt.legend()

plt.show()


