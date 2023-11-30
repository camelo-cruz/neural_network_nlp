import numpy as np
import pandas as pd
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
        
        self.weights = pd.DataFrame([np.random.uniform(-1, 1, size=self.input_size+1)
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
        return None
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
        pass
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
        pass
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
    pass
    #######################################################################