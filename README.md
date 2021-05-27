# Classifying Toxicity
### Text classification with CNN and grid search
**This self assigned project was developed as part of the spring 2021 elective course Cultural Data Science - Language Analytics at Aarhus University.** <br>

__Task:__ This project aims at detecting toxic comments and distinguish these from comments that are clean from offensive language. To find the best model parameters, grid search is employed. 
From this, a model is chosen and used to create a command line tool that can detect toxicity in a comment string.

The data for the project is from a Kaggle competition where they use comments from Wikipedia’s talk page edits to improve online discussions. 
The data can be retrieved from the link: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data. For this project, only the train.csv is used as this contains the labelled data.
The dataset has been labelled by human annotators and they have included multiple labels for different types of toxicity such as ‘threat’, ‘insult’ and ‘identity hate’. 
Contrary to the competition task, I decide to treat the classification as a binary problem. Moreover, I want to implement grid search to find the best hyperparameters for the model.
Finally, the project should end with a command line tool that can take a comment string and classify it as either ‘clean’ or ‘toxic’ using the best model from the grid search.

The outputs of the scripts are provided in the out folder. In this, two subfolder have been created to distinguish between the outputs from the two scripts. 
In the out folder there is also an image of the word cloud created with the clean comments. I decided not to include the word cloud of toxic words as it was extremely depressing to look at. 
However, if the toxic word cloud is desired, it can be obtained with a small modification in the word cloud function in the toxicity-grid-search.py.

The folder grid_search contains the model architecture (as png and txt) and model history (as png) from all the models in the grid search.  
The folder best_model contains the model architecture (as png and txt) and model history (as png) from the model used for the command line tool.

The scripts toxicity-grid-search.py and predict-toxicity.py are in the src and they can run without any input but several parameters can be defined. <br>

### Method: <br>
For this project, I use data from a Kaggle competition on classifying toxic comments. The data is structured so that there are six labels under the toxic classification. 
These labels overlap meaning that the same comment can be labelled as belonging to more than one class. However, I decided to treat this as a binary problem and gather all toxic labels under one class. 
This was also due to the classes within toxicity being very unbalanced. Similarly, the data contain far more comments that are ‘clean’ from toxicity. 
For preprocessing, I therefore start by balancing the data and create a new data frame with equal number of ‘clean’ and ‘toxic’ comments. <br>

To have more control over the tokenization and tidying of the comments, I define my own helper function to tokenize the data and remove stop words as well as other words which I consider as adding noise to the data. 
However, it is not an exhaustive list and given more time the function can definitely be improved by adding more words to the removal list. <br>

To prevent the model overfitting on the training data I employ various methods in the grid search. These methods include trying out two different regularization methods: the lasso and the ridge regularization. 
Both regularization methods are run with the default 0.01 penalty. Furthermore, I try different numbers of nodes in the hidden fully connected layer i.e., 32 and 64 nodes. 
In the model architecture, I also add a max pooling layer instead of a flattening layer and a dropout layer between the fully connected layer and the output layer. 
Lastly, in finding the best hyperparameters I run the models with Adam and Stochastic gradient descent, respectively. 
As many of the words in the corpus are misspelled, I decided to incorporate word embeddings. The embeddings take into account the collocates of the words and this can therefore provide more insightful meaning to the model than simply using a count vectorizer. <br>

For the command line tool for comment prediction I employ the best model from the grid search. I then use this model to predict the label for the comment string provided by the user.




__Dependencies:__ <br>
To ensure the rest of the dependencies are in accordance with the ones used for the script, you can create the virtual environment ‘toxic_venv’ from the command line by executing the bash script ‘create_toxic_venv.sh’. **If using UCloud, you might have to replace line 15 in the bash script with the command ‘sudo apt install graphviz’.**
```
    $ bash ./create_toxic_venv.sh
```
This will install an interactive command-line terminal for Python and Jupyter as well as all packages specified in the ‘requirements.txt’ in a virtual environment. 
After creating the environment, it will have to be activated before running the classification scripts.
```    
    $ source toxic_venv/bin/activate
```
After running these two lines of code, the user can commence running the scripts. <br>

### How to run toxicity-grid-search.py <br>

This script uses TensorFlow Keras and a path to a csv of labelled comments from Wikipedia’s talk page edits to find the optimal parameters for a convolutional neural network that can predict toxicity in a comment. 
If using the same folder structure as in the Github repository, the script toxicity-grid-search.py can run from command line without additional input. 
However, the user can specify the file path of the csv file with the comments. Similarly, the user can also specify test size and number of epochs. 
The default is a test size of 0.25 and 20 epochs. 
The outputs will be a summary of the model architecture (as a txt and a png file) and a plot of the model history (as a png file).  <br>

__Parameters:__ <br>
```
    filepath: str <filepath-of-csv-file>, default = '../data/train.csv'
    test_size: int <test-size>, default = 0.25
    epochs: int <number-of-epochs>, default = 20

```
    
__Usage:__ <br>
```
    toxicity-grid-search.py -f <filepath> -t <test-size> -e <epochs> 
```
    
__Example:__ <br>
```
    $ cd src
    $ python3 toxicity-grid-search.py -f ../data/train.csv -t 0.25 -e 20

```


### How to run predict-toxicity <br>

This script trains a convolutional neural network on comments from Wikipedia’s talk page edits to classify these as either toxic or clean (non-toxic). 
It takes as input a path to the labelled data (train.csv) and a string for toxicity prediction. 
The user can also specify batch size, test size and number of epochs (for this model a descent accuracy is obtained after just 2 epochs). 
The model output can be seen in the 'out' folder as well as in the terminal. The output consists of a png and txt of the model architecture and a png of the model history. 
Moreover, an accuracy score for training and testing data is generated and printed in the terminal. 
Lastly, a prediction for the unseen comment is printed in the terminal.  <br>

__Parameters:__ <br>
```
    filepath: str <filepath-of-csv-file>, default = '../data/train.csv'
    comment: str <comment-for-prediction>, default = 'You make a valid point, but I disagree'
    batch_size: int <batch-size>, default = 128
    test_size: int <test-size>, default = 0.25
    epochs: int <number-of-epochs>, default = 20

```
    
__Usage:__ <br>
```
    predict-toxicity.py -f <filepath> -c <comment> -b <batch-size> -t <test-size> -e <epochs>
```
    
__Example:__ <br>
```
    $ cd src
    $ python3 predict-toxicity.py -f ../data/train.csv -c 'you make a valid point, but I disagree' -b 128 -t 0.25 -e 10

```
The code has been developed and tested on UCloud in the Coder Python application. It has also been tested in Jupyter Hub on worker02. 

### Results and Discussion:
The results of the grid search showed that the models with Adam optimization performed remarkably better than the models using stochastic gradient descent. 
The latter models had an accuracy around 50-60% on the testing data which is only marginally better than chance. In contrast, all models with Adam had an accuracy of above 91%. 
Focusing on the models with Adam optimization, model performance was not greatly affected by regularization method nor the number of nodes in the hidden layer. 
Contrarily, the testing accuracy was in the range of 91.76 – 92.20% across the four models. 
The regularization parameter had a bigger effect on the models with SGD, where using an L2 regularizer improved the model by around 13%. 
So, the best performing model had Adam optimization, L1 regularization and 32 nodes in the hidden fully connected layer. 
This model had an accuracy of 99.33% on the training data and 92.20% on the testing data. Of the models with Adam it had the lowest accuracy on the training data but highest accuracy on the testing data. 
Moreover, the model history plot shows that the model doesn’t seem to overfit on the training data as accuracies on training and testing data seem to be approximately equal (with accuracy on training data being a bit superior). 
When running the best model again outside of the grid search, it performs very poorly. Running it for 20 epochs, the model performance was equal to chance. 
I ran the model again but this time with L2 regularization and 64 nodes. This significantly improved the model and the accuracy increased to 99.91% on the training data and 91.44% on the testing data. 
However, this shows that the models in the grid search are very sensitive to the train test split. 
For further development, the grid search could therefore benefit from cross validation to ensure that the model is robust across splits of the data. <br>
Evaluating the model using Adam optimization, ridge regularization and 64 nodes, it makes sense that the model is so good at predicting toxicity because most of the toxic comments are associated with the same harmful and negative words like “fuck” and “die”. 
The same words are not commonly found in the clean comments (with a few exceptions – see word cloud). 
Looking at the word cloud of clean comments, we see that these comments appeal more to logos and reason and include words such as “claim”, “discussion” and “source”. 

The plot below shows the model history for the CNN run with Adam optimizer, 64 nodes and L2 (ridge) regularization for 20 epochs: <br>
![alt text](https://github.com/miemartinez/ToxicCommentsClassification/blob/main/out/best_model/best_model_history.png?raw=true) <br>

When inspecting the model history plot is seems that the model is overfitting on the training data after the first epoch as the validation loss start to increase while the training loss decreases. 
Similarly, the validation accuracy appears to be stable across epochs while the training accuracy gradually increase and approach 100%. <br>
Considering the preprocessing steps, I did a quite extensive text cleaning but as I am using word embeddings for my model, it might make more sense to not stem the words and remove all punctuation. <br>
In conclusion, the resulting command line tool seems to be working reasonably well though it is slightly overfitted. 
I have tested it on a few sentences, and all have been returned with the intended label (one of which is used as default when running the script).
