#Practice Computer Vision on a Set of Digits
#Created by: Ethan Reese
#Jan 31, 2019
from __future__ import print_function

import glob
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from IPython import display
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import log_loss, accuracy_score

#Make the features into tensorflow columns
def construct_feature_columns():

    return set([tf.feature_column.numeric_column('pixels', shape=784)])
#Seperates the data and returns all of the features as well as the labels
def parse_data(data):
    labels = data['label']
    #Pull out the features and scale from [0,1]
    features = data.iloc[:,1:784]
    features = features/255
    return labels, features
#Make a relevant input function for training given features, labels and batch size
def create_training_input_fn(features, labels, batch_size, num_epochs = None, shuffle = True):
    
    #Create the function to return
    def input_fn_(num_epochs=None, shuffle=True):
        raw_features = {"pixel": features.values}
        raw_targets = np.array(labels)

        
        ds = tf.data.Dataset.from_tensor_slices((dict(raw_features), raw_targets))
        ds = ds.batch(batch_size).repeat(num_epochs)

        if shuffle:
            ds = ds.shuffle(10000)
        
        return ds.make_one_shot_iterator().get_next()
    return input_fn_
#Create an input function to use with the testing data
def create_predict_input_fn(features, labels, batch_size):
    #returns the features and labels for the predictor
    def input_fn_():
        raw_features = {"pixels": features.values}
        raw_targets = np.array(labels)

        ds = tf.data.Dataset.from_tensor_slices((raw_features, raw_targets))
        ds = ds.batch(batch_size)
        
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        print(feature_batch, label_batch)
        return tf.convert_to_tensor(feature_batch), tf.convert_to_tensor(label_batch)
    return input_fn_

#Function used to train neural network
def train_nn_classification(
    learning_rate,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):

    periods = 10
    steps_per_period = steps/periods
    predict_training_input_fn = create_predict_input_fn(training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(validation_examples, validation_targets, batch_size)
    training_input_fn = create_training_input_fn(training_examples, training_targets, batch_size)

    #Create a the relevant classifier
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=construct_feature_columns(),
        n_classes=10,
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        config= tf.estimator.RunConfig(keep_checkpoint_max=1)
    )

    #Make the model run within a loop such that each step of the training is visible

    print("Training Model...")
    print("Logloss Error (on validation set)")
    training_errors = []
    validation_errors = []
    for period in range (0, periods):
        classifier.train(
            input_fn = training_input_fn, 
            steps = steps_per_period)

        #Compute the probabilities for training set
        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilites'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id,10)

        #Compute the probabilities for the validation set
        validation_predictions = list(classifier.predict(input_fn = predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)

        #Compute the errors
        training_log_loss = log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = log_loss(validation_targets, validation_pred_one_hot)

        print(" period %02d, %.02f" % (period, validation_log_loss))

        #Add the errors to the error array
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print("Model Trained")

    #Make the predictions
    final_predictions = classifier.predict(input_fn = predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

    accuracy = accuracy_score(validation_targets, final_predictions)
    print("Accuracy on validation data: %0.2f" % accuracy)

    #Graph out the error
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()

    #Output the confusion matrix
    cm = tf.confusion_matrix(validation_targets, final_predictions)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    return classifier    

#Take the csv file in and turn into dataframe with randomized order
data = pd.read_csv('/Users/Ethan/Devel/Python/MNIST/data/train.csv')
data = data.reindex(np.random.permutation(data.index))
#print(data.describe())

#Take the data and make train and validation  and test pools
training_labels, training_features = parse_data(data[:100])
#print(training_features.describe())

validation_labels, validation_features = parse_data(data[100:120])
#print(validation_features.describe())

test_labels, test_features = parse_data(data[120:130])


classifier = train_nn_classification(
    learning_rate = 0.05,
    steps = 1000,
    batch_size = 30,
    hidden_units = [100,100],
    training_examples = training_features,
    training_targets = training_labels,
    validation_examples = validation_features,
    validation_targets = validation_labels
)

