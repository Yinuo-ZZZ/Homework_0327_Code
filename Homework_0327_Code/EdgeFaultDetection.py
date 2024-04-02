#!/usr/bin/env python
# coding: utf-8

# In[13]:


# load data
import pandas as pd

# raw string to process path
file_path = r'E:\SA\Yinuo Zhang\Homework\Umea University\0327\faults_data.csv'
# load
fault_data = pd.read_csv(file_path, sep=';')


# In[14]:


# data preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# identify the feature

categorical_features = fault_data.select_dtypes(include=['object']).columns
numeric_features = fault_data.select_dtypes(include=['int64', 'float64']).columns

# whether all feature are under fully considered
assert len(categorical_features) + len(numeric_features) == 34, "Total features do not match."

# create the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # process data features' missing values
            ('scaler', StandardScaler())]), numeric_features),  # standardizing
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # process cat features' missing values
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features),  # onehot encoder cat features
        ],
    remainder='passthrough')


# In[15]:


# spilit the dataset, one part is for local device, the other one part is for global 
import numpy as np
from sklearn.model_selection import train_test_split

fault_data = preprocessor.fit_transform(fault_data)
data_local, data_global = train_test_split(fault_data, test_size=0.2, random_state=42)


# In[18]:


# implement the Dirichlet Distribution for three local devices: in Non-IID fashion
num_devices = 3

# Concentration parameter
alpha = 0.5

# randomly set proportion for local devices based on Concentration parameter
proportions = np.random.dirichlet([alpha] * num_devices)

# compute the amount of sample for every devices
num_data_per_device = np.floor(proportions * data_local.shape[0]).astype(int)

# generate and shuffle index for local databse
indices = np.arange(data_local.shape[0])
np.random.shuffle(indices)

# distribute the local dataset to three different local devices
device_data_indices = np.split(indices[:num_data_per_device.sum()], np.cumsum(num_data_per_device)[:-1])

device_data = [data_local[device_indices, :] for device_indices in device_data_indices]

# local dataset for training and testing
local_train_data = []
local_test_data = []
for device_data_i in device_data:
    data_train, data_test = train_test_split(device_data_i, test_size=0.2, random_state=42)
    local_train_data.append(data_train)
    local_test_data.append(data_test)


# In[19]:


# define autoencoder as deep learning method
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

# package the autoencoder for both local and global level
def build_autoencoder(data):
    input_dim = data.shape[1]
    
    input_layer = Input(shape=(input_dim,))
    # Encoding layer
    encoded = Dense(16, activation='relu')(input_layer)
    # Decoding Layer
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    # Encoder Model
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error') #loss function
    
    return autoencoder


# In[26]:


# build a edge deployable deep learning model 
def train_autoencoder(data, epochs=50, batch_size=32, shuffle=True, validation_split=0.2): # package the training process
    
    data = data.toarray() # dense format
    autoencoder = build_autoencoder(data)
    autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2)
    return autoencoder


# train autoencoder database for every device
autoencoders = [train_autoencoder(data, epochs=100, batch_size=32) for data in local_train_data]


# In[27]:


# Fault Detection Process
import matplotlib.pyplot as plt

# define the fault detection function
def detect_faults(autoencoder, data):
    # ensure data is a dense format
    if isinstance(data, np.ndarray):
        pass  # already if numpy
    else:
        data = data.toarray()  # convert sparse matrix to dense format
        
    # reconstruct local test data using autocoder
    reconstructed_data = autoencoder.predict(data)
    # compute reconstruction difference
    mse = np.mean(np.square(data - reconstructed_data), axis=1)
    return mse

# define visualization function
def plot_reconstruction_error_distribution(mse, device_id):
    plt.figure(figsize=(10, 6))
    plt.hist(mse, bins=50, alpha=0.7, color='blue')
    plt.title(f"Device {device_id}: Reconstruction Error Distribution")
    plt.xlabel("Mean Squared Error (MSE)")
    plt.ylabel("Frequency")
    plt.show()

# Fault Detection on every device test!
for i, (autoencoder, test_data) in enumerate(zip(autoencoders, local_test_data)):
    mse = detect_faults(autoencoder, test_data)
    print(f"Device {i+1}, mean squared error: {np.mean(mse)}")
    plot_reconstruction_error_distribution(mse, i+1)


# In[28]:


# fault judgement based on threshold value
def identify_potential_faults(mse, percentile=95):
    sorted_mse = np.sort(mse)  # sort
    threshold = np.percentile(sorted_mse, percentile)  # percentile
    print(f"Threshold for potential faults: {threshold}")
    potential_faults = mse > threshold
    return potential_faults, threshold

# identify the number of fault sample for every device
for i, (autoencoder, test_data) in enumerate(zip(autoencoders, local_test_data)):
    mse = detect_faults(autoencoder, test_data)
    potential_faults, threshold = identify_potential_faults(mse)
    print(f"Device {i+1}, Potential faults count: {np.sum(potential_faults)}")


# In[30]:


# global model (edge server): integration

# initialize
data_global_train, data_global_test = train_test_split(data_global, test_size=0.2, random_state=42)

# train process
train_autoencoder(data_global_train)

# test
global_test_mse = detect_faults(train_autoencoder(data_global_train), data_global_test)

# compute the global testing error distribution
plot_reconstruction_error_distribution(global_test_mse, "Global Model")

# identify potential faults for global
potential_faults_global, _ = identify_potential_faults(global_test_mse)
print(f"Global model, Potential faults count: {np.sum(potential_faults_global)}")


# In[ ]:




