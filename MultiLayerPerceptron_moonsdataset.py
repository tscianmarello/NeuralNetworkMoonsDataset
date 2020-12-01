#Following imports are necessary to run the program:

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_circles, make_moons
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch.optim as optim

#The program first creates the moons dataset which consists of 2 values X and y. The point of the program is to
#accurately classify the dataset using a neural network. Once the dataset is created, the train_test_split function
#separates the dataset into training and testing. Once that is done, the variables are converted to tensors. The
#function detect_GPU() detects whether there is a GPU present to speed up the program running. The class
#MultiLayerPerceptron(nn.Module) contains 4 functions. 1st function: def __init__(self): is the initializer that
#linearly transforms the dataset.

#Neural Network of the MultiLayerPerceptron that inherits from the nn.Module class.
class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        # Linear Transformation of dataset. This will be passed to the forward function that will transform the dataset
        # using the torch.sigmoid() function.
        self.layer_1 = nn.Linear(2, 3)
        self.layer_2 = nn.Linear(3, 2)
        self.layer_3 = nn.Linear(2, 2)
        self.layer_4 = nn.Linear(2, 2)

    #Forward function of the neural network that obtains the transformed data from the consturctor and classifies the
    #dataset using the torch.sigmoid() function. Commented out is the torch.tanh() function that can also be used to
    #classify the dataset.
    #Returns x which is the final classification based on the activation functions.

    def forward(self, x):
        x = self.layer_1(x)
        x = torch.sigmoid(x)
        #x = torch.tanh(x)
        x = self.layer_2(x)
        x = torch.sigmoid(x)
        # x = torch.tanh(x)
        x = self.layer_3(x)
        x = torch.sigmoid(x)
        # x = torch.tanh(x)
        x = self.layer_4(x)
        return x

    # Prediction function that predicts the results based on the torch.nn.functional.log_softmax function and then
    # appends the result to a list on whether the result is either 0 or 1.
    # Returns the result as a tensor.

    def predict(self, x):
        #The list that will hold the predicted results.
        result = []
        #Prediction function.
        prediction = torch.nn.functional.log_softmax(x, dim=1)
        #For loop that separates the predicted result into either a 0 or a 1.
        for A in prediction:
            if A[1] < A[0]:
                result.append(0)
            else:
                result.append(1)
        #Returns tensor.
        return torch.tensor(result)

    #Function returns name of model architecture.
    def name(self):
        return "MultiLayerPerceptron"

#Predicts result related to the graph in db that will be used to draw the contour line separating X and y
#dataset = converts the passed dataset to a tensor.
#result = uses predict function to separate X and y so that the contour line can fit in between.
#Returns result used for the contour line.

def pred1(dataset, mlp):
    #print(dataset)
    dataset = torch.from_numpy(dataset).type(torch.FloatTensor)
    result = mlp.predict(dataset)
    return result

#Outputs the decision boundary graph of the predicted value associated with the MultiLayerPerceptron
def db(x, X):
    x_min = X[:, 0].min() - 1
    y_min = X[:, 1].min() - 1
    x_max = X[:, 0].max() + 1
    y_max = X[:, 1].max() + 1
    x_grid = np.arange(x_min, x_max, 0.01)
    y_grid = np.arange(y_min, y_max, 0.01)
    xx, yy = np.meshgrid(x_grid, y_grid)
    decision_boundary = x(np.c_[xx.ravel(), yy.ravel()])
    decision_boundary = decision_boundary.reshape(xx.shape)
    plt.title("Decision Boundary for MLP")
    plt.contourf(xx, yy, decision_boundary, cmap=plt.cm.Dark2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Greens)
    plt.show()


if __name__ == '__main__':
    #Creates moons dataset for both X and y and then plots it in a scatter plot.
    X, y = make_moons(500, noise=0.2, random_state=42)
    plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], s=40, marker='x', c='red')
    plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], s=40, marker='o', c='blue')
    plt.axis('equal')
    #Calling plt.show() will output the scatter plot.
    #plt.show()

    #Train, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)

    #Converts X and y to tensors
    X, y = Variable(torch.from_numpy(X_train)).float(), Variable(torch.from_numpy(y_train)).long()

    #Detects GPUs and then uses them if present.
    GPUS = 1
    DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and GPUS > 0) else "cpu")

    #Learning rate associated with model
    learning_rate = 0.1
    #Instantiates class of MultiLayerPerceptron
    mlp = MultiLayerPerceptron()
    #loss_function works with either MultiMarginLoss() or CrossEntropyLoss()
    loss_function = nn.CrossEntropyLoss()
    #Chosen optimizer for neural network.
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

    #Number of epochs
    epochs = 1000
    #List that will contain the losses accumulated through the neural network's run. Stored as floats.
    current_loss = []
    #total_loss starts at 0 and calculates the total loss of the neural network.
    total_loss = 0
    #For loop that iterates through all the epochs
    for i in range(1, epochs + 1):
        print("Epoch #: ", i)
        #Prediction based on forward function of MultiLayerPerceptron
        y_prediction = mlp.forward(X)
        #Loss function's current loss
        loss = loss_function(y_prediction, y)
        print("Current loss: ", loss.item(), "\n")
        #Saves current loss in the list.
        current_loss.append(loss.item())
        #Zeros the gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #Prints total losses and sums them.
    print("Total Losses: ", sum(map(float, current_loss)), "\n")

    pred = mlp(X)

    pred = pred.detach().numpy()

    #Prints accuracy score based on sklearn's accuracy_score by taking maximum argument.
    print("Accuracy: ", accuracy_score(y, np.argmax(pred, axis=1)))

    #Plots the decision boundary.
    f = lambda x, mlp=mlp: pred1(x, mlp)
    db(f, X)