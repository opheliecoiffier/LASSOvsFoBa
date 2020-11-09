#packages used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import os
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)
from source import train_error_data, test_error_data


##########################
#Import data ionosphere
############################
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data', header=None)
data = np.array(data)

##################################################
#Formation of the x (data) and y (labels) arrays
#--------------------------------------------------

x = data[:,0:33]
y = data[:,34]
for i in range(351):
    if y[i]=='g':
        y[i] = 1
    else : 
        y[i] = 0

###########################################
#Average training and test squared error
#-------------------------------------------
n = 50
J = 10
train_size = 50
test_size = 301

fig = plt.figure() 
train_error_data(n, J, x, y, train_size)
fig.savefig("training_error_ionosphere.pdf")

fig2 = plt.figure()
test_error_data(n, J, x, y, test_size, train_size)
fig2.savefig("test_error_ionosphere.pdf")

#%%
#############################
#import data Boston housing
#############################

data_housing = load_boston()

X = data_housing.data
Y = data_housing.target

###########################################
#Average training and test squared error
#-------------------------------------------
n = 50 #number of repetition
J = 10 #number of sparsity
train_size = 50
test_size = 456

fig3 = plt.figure()
train_error_data(n, J, X, Y, train_size)
fig3.savefig("training_error_housing.pdf")

fig4 = plt.figure()
test_error_data(n, J, X, Y, test_size, train_size)
fig4.savefig("test_error_housing.pdf")
