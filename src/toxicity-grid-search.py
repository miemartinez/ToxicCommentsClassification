#!/usr/bin/env python
"""
This script uses TensorFlow Keras and a filepath to a csv of labelled comments from Wikipedia’s talk page edits 
to find the optimal parameters for a convolutional neural network that can predict toxicity in a comment. 

As I couldn't get tensorboard to work properly with my browser, I implemented a pyplot of the model history so 
one can track the development of the different models and choose the one that has highest accuracy and is least overfitting.

All outputs will be saved in a created folder called "out" located in the project directory. 
Outputs include summary of model architecture and model history for all possible variations of models.

For more information on the model specifications and implementation of grid search see 'grid_search_util.py' in utils folder in the Github repo.

Parameters:
    filepath: str <filepath-of-csv-file>, default = '../data/train.csv'
    test_size: int <test-size>, default = 0.25
    epochs: int <number-of-epochs>, default = 20
Usage:
    toxicity-grid-search.py -f <filepath> -t <test-size> -e <epochs>
Example:
    $ python3 toxicity-grid-search.py -f ../data/train.csv -t 0.25 -e 20
"""

# import required packages
import pandas as pd
import numpy as np
import os
from contextlib import redirect_stdout
import sys
sys.path.append(os.path.join(".."))

# import utility function to perform grid search
from utils.grid_search_util import GridSearch

# plotting tools
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import pydot
import graphviz

# natural language toolkit 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
# download stopwords and Punkt Sentence Tokenizer
nltk.download('stopwords')
set(stopwords.words('english'))
nltk.download('punkt')

# sklearn
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, 
                                     Dropout, Conv1D)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model

# tensorboard
import tensorboard
from tensorboard.plugins.hparams import api as hp

import argparse

# argparse 
ap = argparse.ArgumentParser()
# adding argument
# path to csv
ap.add_argument("-f", "--filepath", 
                default = "../data/train.csv", 
                help= "Path to comments csv-file")
# test size
ap.add_argument("-t", "--test_size", 
                default = 0.25, 
                help= "Percentage of data to use for testing")
# number of epochs
ap.add_argument("-e", "--epochs", 
                default = 20, 
                help= "Number of epochs")

# parsing arguments
args = vars(ap.parse_args())

def main(args):
    # get arguments from argparse
    filepath = args["filepath"]
    test_size = args["test_size"]
    epochs = args["epochs"]
    
    # create output directory
    dir_name = os.path.join("..", "out", "grid_search")
    create_dir(dir_name)
    
    # initiate ToxicComments class object
    toxicity = ToxicComments(filepath, test_size, epochs)
    
    # use class method to load the data
    data, clean_comments = toxicity.load_and_prepare()
    
    # create word cloud to examine clean comments
    toxicity.word_cloud(clean_comments)
    
    # preprocess data and make train test split
    toxicity.get_train_test_data(data)
    
    # perform grid search
    toxicity.CNN_grid_search()
    
    

class ToxicComments:
    '''

    '''
    def __init__(self, filepath, test_size, epochs):
        '''
        Constructing the class object
        '''
        self.filepath = filepath
        self.test_size = test_size
        self.epochs = epochs
        
    def load_and_prepare(self):
        '''
        Loading the input csv as a pandas data frame. 
        Balancing the data to have equal observations of toxic and clean comments by downsampling the clean comments.
        '''
        print("\n[INFO] Loading csv and balancing data")
        
        #importing the dataset
        data = pd.read_csv(self.filepath)
        # dropping the id as it is unnecessary
        data.drop(['id'],axis=1,inplace=True)

        x=data.iloc[:,2:].sum()

        #marking comments without any tags as "clean"
        rowsums=data.iloc[:,2:].sum(axis=1)
        data['clean']=(rowsums==0)

        #count number of entries and entries within labels
        print("\n[INFO] Total comments = ",len(data))
        print("\n[INFO] Total clean comments = ",data['clean'].sum())
        print("\n[INFO] Total tags =",x.sum())

        # subset data for all clean comments
        clean_comments = data[data.clean==True]
        # add 'clean' as label
        clean_comments["label"] = "clean"
        # subset data for all toxic comments
        toxic_comments = data[data.clean==False]
        # add 'toxic' as label
        toxic_comments["label"] = "toxic"
        
        # downsample observations of clean comments (using sklearn resample function)
        clean_downsampled = resample(clean_comments,
                                     replace=False,
                                     n_samples=len(toxic_comments), # resample as many comments as there are toxic comments
                                     random_state=123)
        
        # merge data from both classes together
        data = pd.concat([clean_downsampled, toxic_comments])

        return data, clean_downsampled
    

    def word_cloud(self, clean_comments):
        '''
        Making a word cloud visualization to gain insight into the data.
        Initially, I did one for the toxic comments as well but that just made me lose faith in humanity 
        a little bit so decided not to include it.
        '''
        comment_words = ''
        stopwords = set(STOPWORDS)

        # iterate through the csv file
        for val in clean_comments["comment_text"].values:      
            # convert each value to string
            val = str(val)

            # split the comment into tokens
            tokens = val.split()

            # convert each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()

            # define text from tokens to use for generating the WordCloud
            comment_words += " ".join(tokens)+" "

        # define WordCloud
        wordcloud = WordCloud(width = 800, 
                              height = 800,
                              background_color ='white',
                              stopwords = stopwords,
                              min_font_size = 10).generate(comment_words)

        # define path for saving WordCloud
        path4wordcloud = os.path.join("..", "out", "wordcloud_clean.png")
        # plot the WordCloud image                       
        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        # save WordCloud
        plt.savefig(path4wordcloud)
        print(f"\n[INFO] Word cloud is saved as {path4wordcloud}")

        
    def get_train_test_data(self, data):
        '''
        Clean comments and make into tokens.
        Make train test split 
        '''
        # applying the tidy text function to all data
        data['comment_text_cleaned'] = data['comment_text'].apply(tidy_text)

        # extract comments and labels from dataframe
        comments = data['comment_text_cleaned'].values
        labels = data['label'].values
        
        # train and test split using sklearn
        X_train, X_test, y_train, y_test = train_test_split(comments, 
                                                            labels, 
                                                            test_size=self.test_size, 
                                                            random_state=42)
        
        # vectorize a text corpus using tensorflow keras Tokenizer
        # initialize tokenizer
        tokenizer = Tokenizer(num_words=10000)
        # fit to training data
        tokenizer.fit_on_texts(X_train)

        # tokenized training and test data
        X_train_toks = tokenizer.texts_to_sequences(X_train)
        X_test_toks = tokenizer.texts_to_sequences(X_test)

        # overall vocabulary size
        self.vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
        
        # max length for comments
        maxlen = 0
        # for comments in the full dataset
        for comment in data["comment_text"]:
            # find length of comment
            length = len(comment)
            # if length is greater than max length
            if length > maxlen:
                # add as new max length
                maxlen = length
        
        self.maxlen = maxlen

        # pad training data to max length
        self.X_train = pad_sequences(X_train_toks, 
                                    padding='post', # sequences can be padded "pre" or "post"
                                    maxlen=self.maxlen)
        
        # pad testing data to max length
        self.X_test = pad_sequences(X_test_toks, 
                                   padding='post', 
                                   maxlen=maxlen)
        
        # transform labels to binarized vectors
        lb = LabelBinarizer()
        self.y_train = lb.fit_transform(y_train)
        self.y_test = lb.fit_transform(y_test)
        

    def CNN_grid_search(self):
        '''
        Grid search to find optimal hyperparameters for the model using the grid_search_utils.
        '''
        # using grid_search_utils to define, train and evaluate model and plot model history 
        grid_search = GridSearch(self.X_train, self.y_train, self.X_test, self.y_test, self.vocab_size, self.maxlen, self.epochs)
      
    
    
def tidy_text(text):
    '''
    Helper function for tidying and tokenizing the text.
    '''
    # list of strings to remove from corpus
    remove_words = ["http", "https", "www.+com$", "nt", "cant", "want", "ne", "im", 
                    "ur", "u", "wont", "ct", "^hahaha.+", "^lolol.+", "^cuntcunt.+"]
    
    # split into words
    tokens = word_tokenize(text)
    
    # convert to lower case
    tokens = [w.lower() for w in tokens]
        
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    
    # filter out stop words from nltk
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    
    # filter out words from remove_words
    words = [w for w in words if not w in remove_words]
    
    return words



    
def create_dir(dirName):
    '''
    Helper function for creating directory if it doesn't exist
    '''
    # if the path does not exist
    if not os.path.exists(dirName):
        # make directory
        os.mkdir(dirName)
        print("\n[INFO] Directory " , dirName ,  " Created ")
    else:   
        # print that it already exists
        print("\n[INFO] Directory " , dirName ,  " already exists")


# behavior from terminal        
if __name__ == "__main__":
    main(args)




