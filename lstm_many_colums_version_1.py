
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
# Get some time series data
df = pd.read_csv("data/NL Pb till Dec 2016.csv")
df.head()


# In[2]:

input_cols = [2, 3, 4, 5, 6, 7]
# Put your inputs into a single list
df['single_input_vector'] = df[input_cols].apply(tuple, axis=1).apply(list)
# Double-encapsulate list so that you can sum it in the next step and keep time steps as separate elements
df['single_input_vector'] = df.single_input_vector.apply(lambda x: [list(x)])
# Use .cumsum() to include previous row vectors in the current row list of vectors
df['cumulative_input_vectors'] = df.single_input_vector.cumsum()


# In[3]:

output_cols = [7]
# If your output is multi-dimensional, you need to capture those dimensions in one object
# If your output is a single dimension, this step may be unnecessary
df['output_vector'] = df[output_cols].apply(tuple, axis=1).apply(list)


# In[4]:

# Pad your sequences so they are the same length
from keras.preprocessing.sequence import pad_sequences

max_sequence_length = df.cumulative_input_vectors.apply(len).max()
# Save it as a list   
padded_sequences = pad_sequences(df.cumulative_input_vectors.tolist(), max_sequence_length).tolist()
df['padded_input_vectors'] = pd.Series(padded_sequences).apply(np.asarray)


# In[5]:

# Extract your training data
X_train_init = np.asarray(df.padded_input_vectors)
# Use hstack to and reshape to make the inputs a 3d vector
X_train = np.hstack(X_train_init).reshape(len(df),max_sequence_length,len(input_cols))
y_train = np.hstack(np.asarray(df.output_vector)).reshape(len(df),len(output_cols))


# In[6]:

# Get your input dimensions
# Input length is the length for one input sequence (i.e. the number of rows for your sample)
# Input dim is the number of dimensions in one input vector (i.e. number of input columns)
input_length = X_train.shape[1]
input_dim = X_train.shape[2]
# Output dimensions is the shape of a single output vector
# In this case it's just 1, but it could be more
output_dim = len(y_train[0])


# In[7]:

from keras.models import Model, Sequential
from keras.layers import LSTM, Dense

# Build the model
model = Sequential()

# I arbitrarily picked the output dimensions as 4
model.add(LSTM(4, input_dim = input_dim, input_length = input_length))
# The max output value is > 1 so relu is used as final activation.
model.add(Dense(output_dim, activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])


# In[8]:

# Set batch_size to 7 to show that it doesn't have to be a factor or multiple of your sample size
history = model.fit(X_train, y_train,
              batch_size=7, nb_epoch=3,
              verbose = 1)


# In[10]:

model.predict(np.array([[[1, 12, 2016, 48, 4, 1735]]]))

