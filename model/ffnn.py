import numpy as np
import numpy.typing as npt

from model.model_utils import softmax, relu, relu_prime
from typing import Tuple


class NeuralNetwork(object):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int, 
        num_classes: int,
        seed: int = 1
    ):
        """
        Initialize neural network's weights and biases.
        """

        ############################# STUDENT SOLUTION ####################
        # YOUR CODE HERE
        #     TODO:
        #         1) Set a seed so that your model is reproducible
        #         2) Initialize weight matrices and biases with uniform
        #         distribution in the range (-1, 1).
        self.input_size= input_size
        self.hidden_size=hidden_size
        self.num_classes=num_classes
        self.seed = seed
        
        np.random.seed(self.seed)
        
        self.W1 = np.array([np.random.uniform(-1, 1, size=self.input_size)
                        for i in range(self.hidden_size)])
        
        self.W_out = np.array([np.random.uniform(-1, 1, size=self.hidden_size)
                        for i in range(self.num_classes)])
        ###################################################################

    def forward(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Forward pass with X as input matrix, returning the model prediction
        Y_hat.
        """
        ######################### STUDENT SOLUTION #########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Perform only a forward pass with X as input.
        
        # index_list = list(X.index)
        # index_list.insert(0, 'bias')
        # X = X.reindex(index_list)
        # X.loc['bias'] = 1
        
        z = np.dot(self.W1, X)
        H = relu(z)
        
        z_1 = np.dot(self.W_out, H)
        O = softmax(z_1)
        
        
        return H, O
        #####################################################################

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Create a prediction matrix with `self.forward()`
        """
        ######################### STUDENT SOLUTION ###########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Create a prediction matrix of the intent data using
        #         `self.forward()` function. The shape of prediction matrix
        #         should be similar to label matrix produced with
        #         `labels_matrix()`
        
        H, output = self.forward(X)
        predictions = np.zeros((self.num_classes, output.shape[1]))
        for column in range(output.shape[1]):
            index = np.argmax(output[:, column])
            one_hot_prediction = np.eye(self.num_classes)[index]
            predictions[:, column]= one_hot_prediction
            
        return predictions
        ######################################################################

    def backward(
        self, 
        X: npt.ArrayLike, 
        Y: npt.ArrayLike
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """
        Backpropagation algorithm.
        """
        ########################## STUDENT SOLUTION ###########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Perform forward pass, then backpropagation
        #         to get gradient for weight matrices and biases
        #         2) Return the gradient for weight matrices and biases
        
        
        m = X.shape[1]
        
        H, O = self.forward(X)
        d_L = O - Y
        dW2 = np.dot(d_L, H.T) / m
        d_l = np.dot(self.W_out.T, d_L) * relu_prime(H)
        dW1 = np.dot(d_l, X.T) / m
        
        return dW1, dW2

        #######################################################################


def compute_loss(pred: npt.ArrayLike, truth: npt.ArrayLike) -> float:
    """
    Compute the cross entropy loss.
    """
    ########################## STUDENT SOLUTION ###########################
    # YOUR CODE HERE
    #     TODO:
    #         1) Compute the cross entropy loss between your model prediction
    #         and the ground truth.
    return -np.sum(truth * np.log(pred), axis=0).mean()
    #######################################################################