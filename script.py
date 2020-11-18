#packages used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import os
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)
from source import train_error_data, test_error_data, alpha_choice, alpha_choice_fig


##########################
# Import data ionosphere
############################
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data', header=None)
data = np.array(data)

##################################################
# Formation of the x (data) and y (labels) arrays
##################################################

x = data[:, 0:33]
y = data[:, 34]
for i in range(351):
    if y[i] == 'g':
        y[i] = 1
    else:
        y[i] = 0

###########################################
# Average training and test squared error
n = 50
J = 10
train_size = 50
test_size = 301
my_alphas = np.array([0.004, 0.006, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4])


fig = plt.figure()
train_error_data(n, J, x, y, train_size, nb_features=33, my_alphas=my_alphas)
fig.savefig("training_error_ionosphere.pdf")


fig2 = plt.figure()
test_error_data(n, J, x, y, test_size, train_size, nb_features=33, my_alphas=my_alphas)
fig2.savefig("test_error_ionosphere.pdf")

####################################
# Choice of alpha for Lasso method

fig_alpha1 = plt.figure()
alpha_choice_fig(x, y, my_alphas, nb_features=33, train_size=train_size)
fig_alpha1.savefig('alpha_choice_lasso.pdf')
print(alpha_choice(x, y, my_alphas, nb_features=33, train_size=train_size))


#############################
# Import data Boston housing
#############################

data_housing = load_boston()

X = data_housing.data
Y = data_housing.target

###########################################
# Average training and test squared error
n = 50  # number of repetition
J = 10  # number of sparsity
train_size = 50
test_size = 456
my_alphas_house = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])

# Training points
fig3 = plt.figure()
train_error_data(n, J, X, Y, train_size, nb_features=13, my_alphas=my_alphas_house)
fig3.savefig("training_error_housing.pdf")

# Test points
fig4 = plt.figure()
test_error_data(n, J, X, Y, test_size, train_size, nb_features=13, my_alphas=my_alphas_house)
fig4.savefig("test_error_housing.pdf")

####################################
# Choice of alpha for Lasso method

fig_alpha2 = plt.figure()
alpha_choice_fig(X, Y, my_alphas_house, nb_features=13, train_size=train_size)
fig_alpha2.savefig('alpha_choice_lasso_house.pdf')
print(alpha_choice(X, Y, my_alphas_house, nb_features=13, train_size=train_size))
