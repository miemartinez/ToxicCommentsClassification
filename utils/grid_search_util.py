#!/usr/bin/env python
"""
Utility script for hyperparameter tuning.

Makes a class object that holds training and testing data, vocabulary size, max length of string input, number of epochs and log directory.

Define model architecture, train and evaluate several models with changing model parameters:

number_nodes = [32, 64]
optimizer = ['sgd', 'adam']
regularizer = ['l1', 'l2']
                      

Saves model architecture and model history recording training loss, validation loss, training accuracy and validation accuracy.

OBS: Outputs are saved in ../out/grid_search. This directory should be made before running the utility script.
"""
# libraries
# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, 
                                     Dropout, Conv1D)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model

# tensorboard
import tensorboard
from tensorboard.plugins.hparams import api as hp

# sklearn
from sklearn.metrics import classification_report


# plotting tools
import matplotlib.pyplot as plt
import pydot
import graphviz
import numpy as np

# system tools
import os
from contextlib import redirect_stdout



# define class for grid search
class GridSearch:
    '''
    Grid Search object.
    Takes training data, validation data, image size and epochs as inputs.
    Defines hyperparameters, performs grid search and saves logs of hyperparameter tuning.
    '''
    def __init__(self, X_train, y_train, X_test, y_test, vocab_size, maxlen, epochs):
        
        # defining changeable hyperparameters 
        HP_NUM_UNITS = hp.HParam('number_nodes', hp.Discrete([32, 64]))
        HP_REGULARIZER = hp.HParam('regularizer', hp.Discrete(['l1', 'l2']))
        HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['Adam', 'SGD']))

        # defining metrics to be recorded
        METRIC_ACCURACY = 'accuracy'

        # create file writer with hyperparameter configuration
        with tf.summary.create_file_writer('../logs/hparam_tuning').as_default():
            hp.hparams_config(
                hparams=[HP_NUM_UNITS, HP_REGULARIZER, HP_OPTIMIZER],
                metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
              )
        
        # training data
        self.X_train = X_train
        self.y_train = y_train
        # validation data
        self.X_test = X_test
        self.y_test = y_test
        # number of epochs
        self.epochs = int(epochs)
        # size of vocabulary
        self.vocab_size = vocab_size
        # max length
        self.maxlen = maxlen
        #log directory
        self.log_dir = '../logs/hparam_tuning'
        
        # starting at session 0
        session_num = 0

        # for loop for grid search
        
        # for the values of num_units
        for number_nodes in HP_NUM_UNITS.domain.values:
            # for the values of regularizer
            for regularizer in HP_REGULARIZER.domain.values:
                # for the values of optimizers
                for optimizer in HP_OPTIMIZER.domain.values:
                    # choose as hyperparameters
                    hparams = {
                        HP_NUM_UNITS: number_nodes,
                        HP_REGULARIZER: regularizer,
                        HP_OPTIMIZER: optimizer,
                    }
                    # define run_name as current session
                    run_name = "run-%d" % session_num
                    # define run directory (changing directory at each run to avoid overwriting)
                    self.run_dir = '../logs/hparam_tuning/' + run_name
                    # print start of trial
                    print('--- Starting trial: %s' % run_name)
                    # print model parameters
                    print({h.name: hparams[h] for h in hparams})
                    
                    # save parameters for use in train_test_model and plot_history
                    num_units = hparams[HP_NUM_UNITS]
                    regularizer = hparams[HP_REGULARIZER] 
                    optimizer = hparams[HP_OPTIMIZER]
                    
                    # create summary log for model training
                    with tf.summary.create_file_writer(self.run_dir).as_default():
                        # record the values used in this trial
                        hp.hparams(hparams)  
                        # define, train and evaluate model with current hyperparameters
                        model, history, model_accuracy = train_test_model(self, number_nodes, regularizer, optimizer, hparams)
                        # plot model history 
                        plot_history(self, history, number_nodes, regularizer, optimizer)
                        
                        # create summary log for accuracy
                        tf.summary.scalar(METRIC_ACCURACY, model_accuracy, step=1)
                    
                    # update session number
                    session_num += 1
                


def train_test_model(self, number_nodes, regularizer, optimizer, hparams):
    '''
    Function for training and testing model.
    Define model architecture with specified hyperparameters and save as .txt and .png.
    Train model and save model history.
    Evaluate model on validation data.
    '''
    # Model arcitecture
    # Sequential model
    # define embedding size we want to work with
    embedding_dim = 50

    # initialize Sequential model
    model = Sequential()
    # add Embedding layer
    model.add(Embedding(input_dim=self.vocab_size,        # vocab size from Tokenizer()
                        output_dim=embedding_dim,         # user defined embedding size
                        input_length=self.maxlen))             # maxlen of padded docs
    model.add(Conv1D(128,                                 # 128 nodes
                     3,                                   # kernel size = 3
                     activation='relu',                   # ReLU activation
                     kernel_regularizer=regularizer))     # kernel regularizer
    # add max pooling instead of flattening layer
    model.add(GlobalMaxPool1D())
    # Add Dense layer; 10 neurons; ReLU activation
    model.add(Dense(number_nodes, 
                    activation='relu'))
    model.add(Dropout(0.1))
    # Add prediction node; 1 node with sigmoid; approximates Logistic Regression
    model.add(Dense(1, 
                    activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    # save model summary
    model_summary = model.summary()

    # name for saving model summary
    model_name = f"Architecture_{number_nodes}_{regularizer}_{optimizer}.txt" 
    # path for saving model summary
    model_path = os.path.join("..", "out", "grid_search", model_name)
    # Save txt file
    with open(model_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    # name for saving plot
    plot_name = f"Architecture_{number_nodes}_{regularizer}_{optimizer}.png" 
    # path for saving plot
    plot_path = os.path.join("..", "out", "grid_search", plot_name)
    # visualization of model summary plot
    model_plot = tf.keras.utils.plot_model(model,
                            to_file = plot_path,
                            show_shapes=True,
                            show_layer_names=True)
    # print that script has saved model architecture
    print(f"\n[INFO] Model architecture is saved as txt in '{model_path}' and as png in '{plot_path}'.")

    # training the model and saving model history
    print(f"\n[INFO] Training the model with opt = {optimizer}, regularizer = {regularizer} and number of nodes = {number_nodes}")

    history = model.fit(self.X_train, self.y_train,
                        validation_data = (self.X_test, self.y_test),
                        epochs=self.epochs, 
                        verbose = 1,
                        batch_size=128,
                        # defining callbacks (couldn't get tensorboard to work in my browser but might just be my computer and network)
                        callbacks=[tf.keras.callbacks.TensorBoard(self.run_dir, profile_batch=0),  # log metrics
                                   hp.KerasCallback(self.run_dir, hparams)],) 

    # evaluate model with test data and save loss and accuracy
    loss, accuracy = model.evaluate(self.X_test, self.y_test, batch_size=128)
    
    print("\n[INFO] Testing Accuracy:  {:.4f}".format(accuracy))
    
    return model, history, accuracy


def plot_history(self, H, number_nodes, regularizer, optimizer):
    """
    Plotting the loss/accuracy of the model during training and saving this as a png file in the out folder.
    """
    # name for saving output
    figure_name = f"History_{number_nodes}_{regularizer}_{optimizer}.png"
    # path for saving output
    figure_path = os.path.join("..", "out", "grid_search", figure_name)
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

