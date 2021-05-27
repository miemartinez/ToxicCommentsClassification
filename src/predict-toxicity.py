#!/usr/bin/env python
"""
This script trains a convolutional neural network on comments from Wikipedia’s talk page edits to classify these as either toxic or clean (non-toxic).
It takes as input a filepath to the labelled data (train.csv) and a string for toxicity prediction.
The user can also specify batch size, test size and number of epochs (for this model a descent accuracy is obtained after just 2 epochs).
The model output can be seen in the 'out' folder as well as in the terminal
The output consists of a png and txt of the model architecture and a png of the model history. 
Moreover, an accuracy score for training and testing data is generated and printed in the terminal. 
Lastly, a prediction for the unseen comment is printed in the terminal.

Parameters:
    filepath: str <filepath-of-csv-file>, default = '../data/train.csv'
    comment: str <comment-for-prediction>, default = 'You make a valid point, but I disagree'
    batch_size: int <batch-size>, default = 128
    test_size: int <test-size>, default = 0.25
    epochs: int <number-of-epochs>, default = 20
Usage:
    predict-toxicity.py -f <filepath> -c <comment> -b <batch-size> -t <test-size> -e <epochs>
Example:
    $ python3 predict-toxicity.py -f ../data/train.csv -c 'you make a valid point, but I disagree' -b 128 -t 0.25 -e 10
## Task
- Train CNN on comment data from Wikipedia's talk page edits to detect toxicity
- Perform toxicity detection in an unseen comment
"""

# import required packages
import pandas as pd
import numpy as np
import os
from contextlib import redirect_stdout

# natural language toolkit 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
# download stopwords and Punkt Sentence Tokenizer
nltk.download('stopwords')
set(stopwords.words('english'))
nltk.download('punkt')

# plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

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

import argparse

# argparse 
ap = argparse.ArgumentParser()
# adding argument
# path to csv
ap.add_argument("-f", "--filepath", 
                default = "../data/train.csv", 
                help= "Path to comments csv-file")
# comment string
ap.add_argument("-c", "--comment", 
                default = 'You make a valid point, but I disagree', 
                help= 'Comment string for prediction')
# batch size
ap.add_argument("-b", "--batch_size", 
                default = 128, 
                help= 'Batch size')
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
    '''
    Main function:
    - load and prepare data
    - train test split
    - define and train CNN
    - create and save plot of model history
    - predict toxicity in new comment
    '''
    # get data from args
    filepath = args["filepath"]
    comment = args["comment"]
    batch_size = args["batch_size"]
    test_size = args["test_size"]
    epochs = int(args["epochs"])
    
    # create output directory
    dir_name = os.path.join("..", "out", "best_model")
    create_dir(dir_name)
    
    # Initialize class object
    predict_toxicity = PredictToxicity(filepath, comment, batch_size, test_size, epochs)
    
    # load data and prepare for binary classification
    data = predict_toxicity.load_and_prepare()
    
    # train test split
    tokenizer = predict_toxicity.get_train_test_data(data)
    
    # define model architecture and train model
    model, history = predict_toxicity.define_and_train_model()
    
    # plot and save model history
    predict_toxicity.plot_history(history)
    
    # predict toxicity for comment
    predict_toxicity.predict_comment(model, tokenizer)


class PredictToxicity:
    '''
    Class for training a convolutional neural network on comments from Wikipedia and using model to predict toxicity in unseen data.
    '''
    def __init__(self, filepath, comment, batch_size, test_size, epochs):
        '''
        Constructing the class object
        '''
        self.filepath = filepath
        self.comment = comment
        self.batch_size = batch_size
        self.test_size = test_size
        self.epochs = epochs
        
    def load_and_prepare(self):
        '''
        Loading the input csv as a pandas data frame. 
        Balancing the data to have equal observations of toxic and clean comments by downsampling the clean comments.
        '''
        print("\n[INFO] Loading csv and balancing data")
        
        # importing the dataset
        data = pd.read_csv(self.filepath)
        # dropping the id column as it is unnecessary
        data.drop(['id'],axis=1,inplace=True)

        # counting tags for each toxic label
        x = data.iloc[:,2:].sum()

        # marking comments without any tags as "clean"
        rowsums=data.iloc[:,2:].sum(axis=1)
        data['clean']=(rowsums==0)

        #count number of entries and entries within labels
        print("\n[INFO] Total comments = ", len(data))
        print("\n[INFO] Total clean comments = ", data['clean'].sum())
        print("\n[INFO] Total toxic comments =", x.sum())

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

        return data
    
    
    def get_train_test_data(self, data):
        '''
        Clean comments and make into tokens.
        Make train test split.
        Pad to same length.
        Binarize labels.
        '''


        comments = data['comment_text'].values
        labels = data['label'].values
        
        # train and test split using sklearn
        X_train, X_test, y_train, y_test = train_test_split(comments, 
                                                            labels, 
                                                            test_size=self.test_size, 
                                                            random_state=42,
                                                            stratify = labels)
        

        # vectorize a text corpus using tensorflow keras Tokenizer
        # initialize tokenizer
        tokenizer = Tokenizer(num_words=10000)
        # fit to training data
        tokenizer.fit_on_texts(X_train)
        
        # tokenized training and test data
        X_train_toks = tokenizer.texts_to_sequences(X_train)
        X_test_toks = tokenizer.texts_to_sequences(X_test)

        # max length for padding
        self.maxlen = max([len(x) for x in np.array(X_train_toks)])

        # overall vocabulary size
        self.vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
        

        # pad training data to max length
        self.X_train = pad_sequences(X_train_toks, 
                                    padding='post', # sequences can be padded "pre" or "post"
                                    maxlen=self.maxlen)
        
        # pad testing data to max length
        self.X_test = pad_sequences(X_test_toks, 
                                   padding='post', 
                                   maxlen=self.maxlen)
        
        # transform labels to binarized vectors
        lb = LabelBinarizer()
        self.y_train = lb.fit_transform(y_train)
        self.y_test = lb.fit_transform(y_test)

        return tokenizer
    
    def define_and_train_model(self):
        '''
        Function for training and testing model.
        Define model architecture with specified hyperparameters and save as .txt and .png.
        Train model and save model history as png.
        Evaluate model on validation data.
        '''
        # define embedding size to work with
        embedding_dim = 50

        # initialize Sequential model
        model = Sequential()
        # add Embedding layer
        model.add(Embedding(input_dim=self.vocab_size,        # vocab size from Tokenizer()
                            output_dim=embedding_dim,         # user defined embedding size
                            input_length=self.maxlen))        # maxlen of padded comments
        # add convolutional layer
        model.add(Conv1D(128,                                 # Nodes: 128
                         3,                                   # kernel size: 3
                         activation='relu',                   # Activation: ReLU
                         kernel_regularizer='l2'))            # kernel regularizer: Ridge
        
        # add max pooling instead of flattening layer
        model.add(GlobalMaxPool1D())
        
        # add Dense layer
        model.add(Dense(64,                                   # Nodes: 64
                        activation='relu'))                   # Activation: ReLU
        # add Dropout layer
        model.add(Dropout(0.1))
        
        # add prediction node
        model.add(Dense(1,                                    # Node: 1
                        activation='sigmoid'))                # Activation: Sigmoid
        
        
        # compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])
        
        
        # save model summary
        model_summary = model.summary()

        # name for saving model summary
        model_name = "Model_architecture.txt" 
        # path for saving model summary
        model_path = os.path.join("..", "out", "best_model", model_name)
        
        # Save txt file
        with open(model_path, 'w') as f:
            with redirect_stdout(f):
                model.summary()
        
        # name for saving plot
        plot_name = f"Model_architecture.png" 
        # path for saving plot
        plot_path = os.path.join("..", "out", "best_model", plot_name)
        
        # visualization of model summary plot
        model_plot = tf.keras.utils.plot_model(model,
                                to_file = plot_path,
                                show_shapes=True,
                                show_layer_names=True)

        # print that script has saved model architecture
        print(f"\n[INFO] Model architecture is saved as txt in '{model_path}' and as png in '{plot_path}'.")

        
        # training the model and saving model history
        print("\n[INFO] Training the model with opt = 'Adam', regularizer = 'L2' and number of nodes = 64")
        
        # fitting model to training data
        history = model.fit(self.X_train, self.y_train,
                            validation_data = (self.X_test, self.y_test),
                            epochs=self.epochs, 
                            verbose = 1,
                            batch_size=self.batch_size) 
        
        # Evaluating model and printing loss and accuracy for training and testing data
        loss, accuracy = model.evaluate(self.X_train, self.y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(self.X_test, self.y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

        return model, history
    


    def plot_history(self, H):
        """
        Plotting the loss/accuracy of the model during training and saving this as a png file in the out folder.
        """
        # name for saving output
        figure_name = f"best_model_history.png"
        # path for saving output
        figure_path = os.path.join("..", "out", "best_model", figure_name)
        # Visualize performance
        plt.style.use("fivethirtyeight")
        plt.figure()
        plt.plot(np.arange(0, self.epochs), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.epochs), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.epochs), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, self.epochs), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_path)

        # print that script has saved
        print(f"\n[INFO] Loss and accuracy across on training and validation is saved as '{figure_path}'.")


    def predict_comment(self, model, tokenizer):
        '''
        Preprocessing string to fit the model input and predicting toxicity label
        '''
        print(f"\nEvaluating comment: '{self.comment}'")
        
        # cleaning and tokenizing the text
        comment = tidy_text(self.comment)
        
        # tokenize comment 
        comment_toks = tokenizer.texts_to_sequences([comment]) # adding comment as an element in a list
        
        # pad training data to max length
        comment_pad = pad_sequences(comment_toks, 
                                    padding='post', # sequences can be padded "pre" or "post"
                                    maxlen=self.maxlen)
        
        # predict label
        y_pred = model.predict(comment_pad)

        # round to 0 or 1 and save as class
        y_class = y_pred.argmax(axis = 1)

        # if class is 1
        if y_class == 1:
            print("This is a toxic comment! You should ignore this and go on with your day :)")
        # if class i 0
        elif y_class == 0:
            print("This is a clean comment! This person is open to constructive discussion :)")
        else:
            print("Something went wrong... Try again!")
    
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
    
    
    
    
    
    