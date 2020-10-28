from sklearn.linear_model import LassoCV, OrthogonalMatchingPursuitCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
   
#x_train, y_train and x_test, y_test cutting 
resultat = []
resultat_lasso = []
X_train, X_test, y_train, y_test = train_test_split(x, y,train_size=50)

#application with LASSO and OMP models to compare the performance on training values 
for i in range(10):    
    reg = OrthogonalMatchingPursuitCV(cv=5, max_iter=i).fit(X_train, y_train)
    resultat.append((1-reg.score(X_train, y_train))*100)
    reg2 = LassoCV(cv=5, max_iter=i).fit(X_train, y_train)
    resultat_lasso.append((1-reg2.score(X_train, y_train))*100)
    
#plot the training error
plt.plot(resultat, label='OMP')
plt.plot(resultat_lasso, label='Lasso')
plt.xlabel('sparsity')
plt.ylabel('training error en %')
plt.title('Performance comparison on real data')
plt.tight_layout()
plt.legend()