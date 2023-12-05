import numpy as np
import matplotlib.pyplot as plt
from model.ffnn import NeuralNetwork, compute_loss



def batch_train(X, Y, model, train_flag=False, batch_size=None):
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
    
    if not batch_size:
        batch_size = X.shape[1]
            
    if train_flag:
        loss_to_print = []
        acc_to_print = []
        iter_batch = 0
        for i in range(1000):
            for minibatch in create_minibatch(X, Y, batch_size):
                iter_batch += 1
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
            
            loss_to_print.append(loss)
            acc_to_print.append(accuracy)
            
            print(f'epoch {i+1}: model loss {loss}, model accuracy {accuracy}')
    
    return loss_to_print, acc_to_print, iter_batch
    ###############################################################################


def minibatch_train(X, Y, model, train_flag=False):
    ########################## STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) As bonus, train your neural network with batch size = 64
    #         and SGD (batch size = 1) for 1000 epochs using learning rate
    #         = 0.005. Then, plot the cost vs iteration for both cases.
    
    SGD_model = NeuralNetwork(model.input_size, model.hidden_size, 
                              model.num_classes, model.seed)
    
    loss_SGD, acc_SGD, iter_SGD = batch_train(X, Y, SGD_model, 
                                                    train_flag=True, batch_size=1)
    
    if train_flag:
        loss_to_print = []
        acc_to_print = []
        iter_minibatch = 0
        for i in range(1000):
            #Calculating gradient with backpropagation and uploading weights
            #with minibatch
            for minibatch in create_minibatch(X, Y, batch_size=64):
                iter_minibatch += 1
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
            
            loss_to_print.append(loss)
            acc_to_print.append(accuracy)
            
            print(f'epoch {i+1}: model loss {loss}, model accuracy {accuracy}')
        
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        bars = plt.bar(['iterations mini_batch', 'iterations SGD'], [iter_minibatch, iter_SGD])
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
            
        plt.ylabel('Number of Iterations')
        plt.title('Number of Iterations')
        

        plt.subplot(1, 3, 2)
        plt.plot(loss_to_print, label='Mini-Batch Training')
        plt.plot(loss_SGD, label='SGD Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Cost vs Epoch (Mini-Batch Training)')
        plt.legend()
        

        plt.subplot(1, 3, 3)
        plt.plot(acc_to_print, label='Mini-Batch Training')
        plt.plot(acc_SGD, label='SGD Training')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epoch (Mini-Batch Training)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/SGD_minibatch.png')
        plt.show()
        
        
            
    #########################################################################

def create_minibatch(X, Y, batch_size):
    indices = np.arange(X.shape[1])
    np.random.shuffle(indices)
    for i in range(0, X.shape[1] - batch_size + 1, batch_size):
        batch = indices[i: i + batch_size]
        yield X[:, batch], Y[:, batch]