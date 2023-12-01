import numpy as np
from model.ffnn import NeuralNetwork, compute_loss



def batch_train(X, Y, model, train_flag=False):
    ################################# STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) Use your neural network to predict the intent
    #         (without any training) and calculate the accuracy 
    #         of the classifier. Should you be expecting high
    #         numbers yet?
    #         2) if train_flag is true, run the training for 1000 epochs using 
    #         learning rate = 0.005 and use this neural network to predict the 
    #         intent and calculate the accuracy of the classifier
    #         3) Then, plot the cost function for each iteration and
    #         compare the results after training with results before training
    
    #Here I am calculating accuracy with no training at all
    preds = model.predict(X)
    num_columns = Y.shape[1]
    equal_columns = [1 for index in range(num_columns) 
                     if np.all(preds[:, index] == Y[:, index], axis=0)]
    accuracy = np.sum(equal_columns) / num_columns
    
    print(f'accuracy before training {accuracy}')
            
    if train_flag:
        for i in range(1000):
            #Calculating gradient with backpropagation and uploading weights
            dW1, dW2 = model.backward(X, Y)
            model.W1 -= 0.005 * dW1 #for hidden layer
            model.W_out -= 0.005 * dW2 #for ouput layer
            
            
            #Computing metrics for evaluation
            #Computing loss
            H, O = model.forward(X)
            loss = compute_loss(O, Y)
            #Computing accuracy
            preds = model.predict(X)
            
            num_columns = Y.shape[1]
            equal_columns = [1 for index in range(num_columns) 
                             if np.all(preds[:, index] == Y[:, index], axis=0)]
            accuracy = np.sum(equal_columns) / num_columns
            
            print(f'epoch {i+1}: model loss {loss}, model accuracy {accuracy}')
            
    return None
    ###############################################################################


def minibatch_train(X, Y, model, train_flag=False):
    ########################## STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) As bonus, train your neural network with batch size = 64
    #         and SGD (batch size = 1) for 1000 epochs using learning rate
    #         = 0.005. Then, plot the cost vs iteration for both cases.
    if train_flag:
        batch_size = 64
        for i in range(1000):
            #Calculating gradient with backpropagation and uploading weights
            for minibatch in create_minibatch(X, Y, batch_size):
                minibatch_X, minibatch_Y = minibatch
                dW1, dW2 = model.backward(minibatch_X, minibatch_Y)
                model.W1 -= 0.005 * dW1 #for hidden layer
                model.W_out -= 0.005 * dW2 #for ouput layer
                
            #Computing metrics for evaluation
            #Computing loss
            H, O = model.forward(X)
            loss = compute_loss(O, Y)
            #Computing accuracy
            preds = model.predict(X)
            num_columns = Y.shape[1]
            equal_columns = [1 for index in range(num_columns) 
                             if np.all(preds[:, index] == Y[:, index], axis=0)]
            accuracy = np.sum(equal_columns) / num_columns
            print(f'epoch {i+1}: model loss {loss}, model accuracy {accuracy}')
            
    #########################################################################

def create_minibatch(X, Y, batch_size=64):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for i in range(0, X.shape[0] - batch_size + 1, batch_size):
        batch = indices[i: i + batch_size]
        yield X[:, batch], Y[:, batch]