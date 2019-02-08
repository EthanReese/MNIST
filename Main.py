#Practice Computer Vision on a Set of Digits
#Created by: Ethan Reese
#Jan 31, 2019
from __future__ import print_function

import glob
import math
import os

from IPython import display
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.python.data import Dataset

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

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))
        ds = ds.batch(batch_size).repeat(num_epochs)

        if shuffle:
            ds = ds.shuffle(10000)
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch
#Create an input function to use with the testing data
def create_predict_input_fn(features, labels, batch_size):
    #returns the features and labels for the predictor
    def input_fn_():
        raw_features = {"pixels": features.values}
        raw_targets = np.array(labels)

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))
        ds = ds.batch(batch_size)

        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch
    return input_fn_

#Function used to train neural network
def 
#Take the csv file in and turn into dataframe with randomized order
data = pd.read_csv('/Users/Ethan/Devel/Python/MNIST/data/train.csv')
data = data.reindex(np.random.permutation(data.index))
#print(data.describe())

#Take the data and make train and validation  and test pools
training_labels, training_features = parse_data(data[:28000])
#print(training_features.describe())

validation_labels, validation_features = parse_data(data[28000:38000])
#print(validation_features.describe())

test_labels, test_features = parse_data(data[38000:42000])

#Take a random piece of test data and display
example = np.random.choice(training_features.index)
_, ax = plt.subplots()
ax.matshow(training_features.loc[example].values.reshape(28,28))
ax.set_title("Label: %i" % training_labels.loc[example])
ax.grid(False)

