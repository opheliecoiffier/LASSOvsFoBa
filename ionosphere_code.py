#packages used
from sklearn.linear_model import LassoCV, OrthogonalMatchingPursuitCV, LarsCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#import data ionosphere
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data', header=None)
data = np.array(data)


#formation of the x (data) and y (labels) arrays
x = data[:,0:33]
y = data[:,34]
for i in range(351):
    if y[i]=='g':
        y[i] = 1
    else : 
        y[i] = 0

#initialisation of needed arrays
n = 10 #number of repetition
J = 10 #number of sparsity
resu = np.zeros(n*10, dtype='float').reshape(10,n)
resu_OMP = np.zeros(n*10, dtype='float').reshape(10,n)
res_OMP = np.zeros(n*10).reshape(10,n)
res_lasso = np.zeros(n*10).reshape(10,n)
res_MSE_lasso = np.zeros(n)
resu_lars = np.zeros(n*10, dtype='float').reshape(10,n)
res_lars = np.zeros(n*10).reshape(10,n)

#%%
#application with LASSO, OMP and lars models to compare the performance on testing values
#repetition : 10 times
for i in range(n):
    X_train, X_test, y_train, y_test = train_test_split(x, y,train_size=50)
    for j in range(J):
        reg2 = LassoCV(cv=5,max_iter=j+1).fit(X_train, y_train)
        reg = OrthogonalMatchingPursuitCV(cv=5, max_iter=j+1).fit(X_train, y_train)
        reg3 = LarsCV(cv=5, max_iter=j+1).fit(X_train, y_train)
        resu_lars[j,i] = 1-reg3.score(X_test, y_test)
        res_lars[j,i] = (y_test[j]-reg3.predict(X_test)[j])**2
        resu[j,i] = 1-reg2.score(X_test, y_test)
        resu_OMP[j,i] = 1-reg.score(X_test, y_test)
        res_OMP[j,i] = (y_test[j]-reg.predict(X_test)[j])**2
        res_lasso[j,i] = (y_test[j]-reg2.predict(X_test)[j])**2

#calculate the mean of the 10 repetitions
somme2 = np.zeros(n)
somme = np.zeros(n)
somme_OMP = np.zeros(n)
somme2_OMP = np.zeros(n)
somme_lars = np.zeros(n)
somme2_lars = np.zeros(n)
for k in range(n):
    for r in range(J):
        somme[r] = res_lasso[r,k] + somme[r]
        somme_OMP[r] = res_OMP[r,k] + somme_OMP[r]
        somme2[r] = resu[r,k] + somme2[r]
        somme2_OMP[r] = resu_OMP[r,k] + somme2_OMP[r]
        somme_lars[r] = res_lars[r,k] + somme_lars[r]
        somme2_lars[r] = resu_lars[r,k] + somme2_lars[r]

#plot the results
plt.plot(somme/n, label='Lasso : MSE')
plt.plot(somme2/n, label='Lasso : R2')
plt.plot(somme2_OMP/n, label='OMP : R2')
plt.plot(somme_OMP/n, label='OMP : MSE')
plt.plot(somme_lars/n,label='Lars : MSE')
plt.plot(somme2_lars/n, label='Lars : R2')
plt.xlabel('sparsity')
plt.ylabel('test error')
plt.title('Performance comparison on simulation data : ionosphere')
plt.legend()

#%%
#application to compare the performance on training values
for i in range(n):
    X_train, X_test, y_train, y_test = train_test_split(x, y,train_size=50)
    for j in range(J):
        reg2 = LassoCV(cv=5,max_iter=j+1).fit(X_train, y_train)
        reg = OrthogonalMatchingPursuitCV(cv=5, max_iter=j+1).fit(X_train, y_train)
        reg3 = LarsCV(cv=5, max_iter=j+1).fit(X_train, y_train)
        resu_lars[j,i] = 1-reg3.score(X_train, y_train)
        res_lars[j,i] = (y_train[j]-reg3.predict(X_train)[j])**2
        resu[j,i] = 1-reg2.score(X_train, y_train)
        resu_OMP[j,i] = 1-reg.score(X_train, y_train)
        res_OMP[j,i] = (y_train[j]-reg.predict(X_train)[j])**2
        res_lasso[j,i] = (y_train[j]-reg2.predict(X_train)[j])**2

#calculate the mean of the 10 repetitions
somme2 = np.zeros(n)
somme = np.zeros(n)
somme_OMP = np.zeros(n)
somme2_OMP = np.zeros(n)
somme_lars = np.zeros(n)
somme2_lars = np.zeros(n)
for k in range(n):
    for r in range(J):
        somme[r] = res_lasso[r,k] + somme[r]
        somme_OMP[r] = res_OMP[r,k] + somme_OMP[r]
        somme2[r] = resu[r,k] + somme2[r]
        somme2_OMP[r] = resu_OMP[r,k] + somme2_OMP[r]
        somme_lars[r] = res_lars[r,k] + somme_lars[r]
        somme2_lars[r] = resu_lars[r,k] + somme2_lars[r]

#plot the results
plt.plot(somme/n, label='Lasso : MSE')
plt.plot(somme2/n, label='Lasso : R2')
plt.plot(somme2_OMP/n, label='OMP : R2')
plt.plot(somme_OMP/n, label='OMP : MSE')
plt.plot(somme_lars/n,label='Lars : MSE')
plt.plot(somme2_lars/n, label='Lars : R2')
plt.xlabel('sparsity')
plt.ylabel('training error')
plt.title('Performance comparison on simulation data : ionosphere')
plt.legend()