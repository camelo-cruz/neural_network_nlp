import numpy as np
import numpy.typing as npt
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize

from typing import Tuple, List, Set


def bag_of_words_matrix(sentences: List[str]) -> npt.ArrayLike:
    """
    Convert the dataset into V x M matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    
    #Tokenizing sentences for hanling tokens
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    #Normalizing words for creating a vocabulary
    words = [word.lower() for sentence in tokenized_sentences for word in sentence]
    vocab = Counter(words) #Counting words for UNK preprocessing
    
    #Changing noncommon words with vocab information which I just created
    #I changed tokens in tokenized sentences by enumerating the sentences and
    #Getting them by index from enumeration
    for i, sentence in enumerate(tokenized_sentences):
        tokenized_sentences[i] = ['UNK' if vocab[word] <= 2 
                                  else word for word in sentence]
        
    #Created a modified vocabulary for using it as my indexes in the dataframe
    #and use them for indexing by word and adding countings
    modified_vocab = Counter({'UNK' if count <= 2 else old_key: count 
                              for old_key, count in vocab.items()})
    
    #Creating  bag of words with words in rows and items in columns
    bag_of_words = pd.DataFrame(np.zeros((len(modified_vocab), len(sentences))),
                                index=modified_vocab.keys())
    
    #putting counts
    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            bag_of_words.loc[word][i] += 1 
    
    #bag of words is now numpy array and added 1 for every item in position 0
    #counting as bias
    bag_of_words = np.array(bag_of_words)
    bag_of_words = np.insert(bag_of_words, 0, np.ones(bag_of_words.shape[1]), axis=0)
    
    return bag_of_words
    #########################################################################


def labels_matrix(data: Tuple[List[str], Set[str]]) -> npt.ArrayLike:
    """
    Convert the dataset into K x M matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    items = data[0]
    classes = list(data[1]) #mofied set as list for using position as index for one hot
    
    #matrix has now classes one hot encoded indexing eye matrices by position in 
    #list with unique classes
    matrix = []
    for i, label in enumerate(items):
        matrix.append(np.eye(len(classes))[classes.index(label)])
    
    #transposed matrix for having items in columns and classes in rows 
    matrix = np.array(matrix).T
    
    return matrix
    #########################################################################


def softmax(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Softmax function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    return np.exp(z) / np.sum(np.exp(z), axis=0)
    #########################################################################


def relu(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Rectified Linear Unit function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    return np.maximum(0, z)
    #########################################################################


def relu_prime(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    First derivative of ReLU function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    
    return np.where(z > 0, 1, 0)
    #########################################################################