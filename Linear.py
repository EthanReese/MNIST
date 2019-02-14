#Attempt to use a linear classifier to predict
from __future__ import print_function

import glob
import os
import math
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
    features = data.iloc[1:784]
    features = features/255
    return labels, features
#Make a relevant input function for training given features, labels and batch size
def create_training_input_fn(features, labels, batch_size, num_epochs = None, shuffle = True):
    
    #Create the function to return
    def input_fn_(num_epochs=None, shuffle=True):
        #Shuffle up the data to be safe
        idx = np.random.permutation(features.index)
        raw_features = {"pixels":features.reindex(idx)}
        raw_targets = np.array(labels[idx])

        ds = tf.data.Dataset.from_tensor_slices((raw_features, raw_targets))
        ds = ds.batch(batch_size).repeat(num_epochs)


        if shuffle:
            ds = ds.shuffle(10000)
        
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        print(feature_batch)
        print(label_batch)
        return feature_batch, label_batch
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
        return feature_batch, label_batch
    return input_fn_
#Train a linear classifier
def train_linear_classification_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets
):
    periods = 10

    steps_per_period = steps/periods

    predict_training_input_fn= create_predict_input_fn(training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(validation_examples, validation_targets, batch_size)
    training_input_fn = create_training_input_fn(training_examples, training_targets, batch_size)

    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    
    classifier = tf.estimator.LinearClassifier(
        feature_columns= construct_feature_columns(),
        n_classes=10,
        optimizer=my_optimizer,
        config=tf.estimator.RunConfig(keep_checkpoint_max=1)
    )

    print("Training Model...")
    print("LogLoss Error (on validation data):")
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        classifier.train(
            input_fn = training_input_fn,
            steps = steps_per_period
        )

        #Save the predictions for the training sets and validation sets
        training_predictions = list(classifier.predict(input_fn = predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)

        validation_predictions = list(classifier.predict(input_fn = predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for itme in training_predictions])
        validation_pred_class_id = np.array([item['class_ids'] for item in training_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)

        #Calculate the log loss
        training_log_loss = log_loss(training_targets, training_pred_one_hot)
        validation_los_loss = log_loss(validation_examples, validation_pred_one_hot)
        
        print(" period %02d: %02f" % (period, validation_los_loss))

        training_errors.append(training_log_loss)
        validation_errors.append(validation_los_loss)
    print("Model Trained")

    final_predictions = classifier.predict(input_fn)

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

classifier = train_linear_classification_model(
    learning_rate = 0.02,
    steps = 100,
    batch_size=15,
    training_examples = training_features,
    training_targets = training_labels,
    validation_examples = validation_features,
    validation_targets = validation_labels
)