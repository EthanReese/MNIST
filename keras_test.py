from __future__ import absolute_import, division, print_function

import tensorflow as tf
import pandas as pd
import numpy as np
import talos as ta
from tensorflow.keras import optimizers

def model(x_train, y_train, x_val, y_val, parameters):
    #Build up the Keras model
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation=parameters['activation']),
    tf.keras.layers.Dropout(parameters['dropout'])
      ])
    #Apparently Keras has to finish with a dense layer
    model.add(tf.keras.layers.Dense(10, activation=parameters['last_activation']))

    model.compile(optimizer=parameters['optimizer'], loss = parameters['losses'],
            metrics = ['accuracy'])

    history = model.fit(x_train, y_train,
        validation_data=[x_val, y_val],
        batch_size=parameters['batch_size'],
        epochs=parameters['epochs'])
    return history, model

#Read in the data and convert to numpy array
data = pd.read_csv('/Users/Ethan/Devel/Python/MNIST/data/train.csv')
data = data.reindex(np.random.permutation(data.index))
data = data.values

x_pre = data[1:42001]
x = x_pre[:,1:].reshape(x_pre.shape[0], 28, 28).astype('float32')
x /=255
y_pre = data[1:42001]
y = y_pre[:,0]
#Process the training data into two separate arrays
"""train = data[1:37999]
y_train = train[:,0]
x_train = train[:,1:].reshape(train.shape[0], 28, 28).astype('float32')
x_train = x_train/255

#Process the data for testing into two separate arrays
test = data[38000:42001]
y_test = test[:,0]
x_test = test[:,1:].reshape(test.shape[0], 28, 28).astype('float32')
x_test = x_test/255"""

#Make the parameter evaluation set
p = {'lr': (0.5, 5, 10),
     'first_neuron':[4, 8, 16, 32, 64],
     'batch_size': (2, 30, 10),
     'epochs': [40],
     'dropout': (0, 0.5, 5),
     'weight_regulizer':[None],
     'emb_output_dims': [None],
     'shape':['brick','long_funnel'],
     'optimizer': ['Adam', 'Nadam', 'RMSprop'],
     'losses': ['logcosh'],
     'activation':['relu', 'elu'],
     'last_activation': ['sigmoid']}

t = ta.Scan(x=x,
            y=y,
            model=model,
            grid_downsample=0.01, 
            params=p,
            dataset_name='MNIST',
            experiment_no='1')

#model.evaluate(x_test, y_test, steps=30)