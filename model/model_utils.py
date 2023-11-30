import numpy as np
import numpy.typing as npt
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize

from typing import Tuple, List, Set


def bag_of_words_matrix(sentences: List[str]) -> pd.DataFrame:
    """
    Convert the dataset into V x M matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    words = [word.lower() for sentence in tokenized_sentences for word in sentence]
    vocab = Counter(words)
    
    for i, sentence in enumerate(tokenized_sentences):
        tokenized_sentences[i] = ['UNK' if vocab[word] <= 2 
                                  else word for word in sentence]
        
    modified_vocab = Counter({'UNK' if count <= 2 else old_key: count 
                              for old_key, count in vocab.items()})
    
    bag_of_words = pd.DataFrame(np.zeros((len(modified_vocab), len(sentences))),
                                index=modified_vocab.keys())
    
    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            bag_of_words.loc[word][i] += 1 
    
    return bag_of_words
    #########################################################################


def labels_matrix(data: Tuple[List[str], Set[str]]) -> pd.DataFrame:
    """
    Convert the dataset into K x M matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    items = data[0]
    classes = data[1]
    
    matrix = pd.DataFrame(np.zeros((len(classes), len(items))),
                                index=classes.keys())
    
    for i, label in enumerate(items):
        matrix.loc[label][i] += 1 
    
    return matrix
    #########################################################################


def softmax(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Softmax function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    return np.exp(z) / np.sum(np.exp(z))
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