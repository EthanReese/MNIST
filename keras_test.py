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
"""def model_final(x_train, y_train, x_val, y_val, parameters):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='elu'),
    tf.keras.layers.Dropout(0)
      ])
    #Apparently Keras has to finish with a dense layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='Adam', loss = parameters['losses'], metrics = ['accuracy'])

    model.fit(x_train, y_train, validation_data = [x_val, y_val], batch_size=21, epochs=150)
    
    return model"""
#Read in the data and convert to numpy array
data = pd.read_csv('/Users/Ethan/Devel/Python/MNIST/data/train.csv')
data = data.reindex(np.random.permutation(data.index))
data = data.values

#Read in the test data and convert into numpy array
test_data = pd.read_csv('/Users/Ethan/Devel/Python/MNIST/data/test.csv')
test_data = test_data.reindex(np.random.permutation(test_data.index))
test_data = test_data.values

x_pre = data[1:37999]
x = x_pre[:,1:].reshape(x_pre.shape[0], 28, 28).astype('float32')
x /=255
y_pre = data[1:37999]
y = y_pre[:,0]
x_val_pre = data[37999:42001]
x_val = x_val_pre[:,1:].reshape(x_val_pre.shape[0], 28, 28).astype('float32')
y_val_pre = data[37999:42001]
y_val = y_val_pre[:,0]


#Process the test data into a normal numpy array
train_x_pre = test_data[1:28001]
train_x = train_x_pre[:,0:].reshape(train_x_pre.shape[0], 28, 28).astype('float32')
train_x /= 255
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
     'first_neuron':[16, 32, 64],
     'hidden_layers': [0,1,2],
     'batch_size': [10, 12, 14, 16, 18, 20, 22, 24, 26],
     'epochs': [50],
     'dropout': [0, 0.3, 0.5, 0.9],
     'weight_regulizer':[None],
     'emb_output_dims': [None],
     'shape':['brick', 'long_funnel'],
     'optimizer': ['Adam', 'Nadam', 'RMSprop'],
     'losses': ['logcosh'],
     'activation': ['relu', 'elu'],
     'last_activation': ['sigmoid']}

t = ta.Scan(x=x,
            y=y,
            model=model,
            grid_downsample=0.01, 
            params=p,
            dataset_name='MNIST')

#begin some analysis of the results
r = ta.Reporting(t)

print("Highest Result: " + r.high())
print("Best Parameters: " + r.best_params())
print("Correlations: " + r.correlate('acc'))

print("Graphs: ")
r.plot_kde('val_acc')
r.plot_line()
r.plot_regs()
r.plot_hist(bins=50)
r.plot_corr()
r.plot_corr()

#Run the model with the test data
#model = model_final(x, y, x_val, y_val, p)

#predictions = model.predict(train_x)

#prediction = pd.DataFrame(predictions, columns=['predictions']).to_csv('prediction.csv')