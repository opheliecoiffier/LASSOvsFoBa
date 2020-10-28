from sklearn.linear_model import LassoCV, OrthogonalMatchingPursuitCV, LarsCV
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
res_OMP = []
res_lasso = []
res_lars = []
resultat_OMP = []
resultat_lasso = []
resultat_FoBa = []
X_train, X_test, y_train, y_test = train_test_split(x, y,train_size=50)

#application with LASSO and OMP models to compare the performance on training values
J = 10
for i in range(1,10): 
    for j in range(1,J):
        reg = OrthogonalMatchingPursuitCV(cv=5, max_iter=i).fit(X_train, y_train)
        reg2 = LassoCV(cv=5, max_iter=i).fit(X_train, y_train)
        reg3 = LarsCV(cv=5, max_iter=i).fit(X_train, y_train)
        res_OMP.append(1-(reg.score(X_train, y_train))**2)
        res_lasso.append(1-(reg2.score(X_train, y_train))**2)
        res_lars.append(1-(reg3.score(X_train, y_train))**2)
        
    resultat_OMP.append(np.mean(res_OMP))
    resultat_lasso.append(np.mean(res_lasso))
    resultat_FoBa.append(np.mean(res_lars))
    
    
#plot the training error
plt.plot(resultat_FoBa, label='FoBa')
plt.plot(resultat_OMP, label='OMP')
plt.plot(resultat_lasso, label='Lasso')
plt.xlabel('sparsity')
plt.ylabel('training error en %')
plt.title('Performance comparison on real data')
plt.tight_layout()
plt.legend()

#%%
#x_train, y_train and x_test, y_test cutting 
resultat = []
resultat_lasso = []
resultat_FoBa = []
X_train, X_test, y_train, y_test = train_test_split(x, y,train_size=50)

#application with LASSO and OMP models to compare the performance on training values 
for i in range(30):    
    reg = OrthogonalMatchingPursuitCV(cv=5, max_iter=i).fit(X_test, y_test)
    resultat.append((1-reg.score(X_train, y_train))*100)
    reg2 = LassoCV(cv=5, max_iter=i).fit(X_test, y_test)
    resultat_lasso.append((1-reg2.score(X_test, y_test))*100)
    reg3 = LarsCV(max_iter=i,cv=5).fit(X_test, y_test)
    resultat_FoBa.append((1-reg3.score(X_test, y_test))*100)
    
#plot the testing error
plt.plot(resultat_FoBa, label='FoBa')
plt.plot(resultat, label='OMP')
plt.plot(resultat_lasso, label='Lasso')
plt.xlabel('sparsity')
plt.ylabel('testing error en %')
plt.title('Performance comparison on real data')
plt.tight_layout()
plt.legend()