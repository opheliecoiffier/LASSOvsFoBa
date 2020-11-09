#packages used
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit, Lars
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

################################
#Average train squared error
################################

def train_error_data(n, J, x, y, train_size):
    #initialisation : 3 methods (Lasso, Larsn OMP)
    vec = np.zeros(train_size*J).reshape(train_size,J)
    res = np.zeros(n*J).reshape(n,J)    
    somme = np.zeros(J)
    vec2 = np.zeros(train_size*J).reshape(train_size,J)
    res2 = np.zeros(n*J).reshape(n,J)
    somme2 = np.zeros(J)
    vec3 = np.zeros(train_size*J).reshape(train_size,J)
    res3 = np.zeros(n*J).reshape(n,J)
    somme3 = np.zeros(J)
    axes = np.arange(1,11)

    #average training squared error : n iterations and sparsity (1 to J)
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(x, y,train_size=train_size)
        for j in range(J):
            reg2 = Lasso(alpha=1/(j+1)).fit(X_train, y_train)
            reg = OrthogonalMatchingPursuit(n_nonzero_coefs=j+1).fit(X_train, y_train)
            reg3 = Lars(n_nonzero_coefs=j+1).fit(X_train, y_train)
            vec[:,j] = (y_train-reg.predict(X_train))**2
            res[i,j] = sum(vec[:,j])/train_size
            vec2[:,j] = (y_train-reg2.predict(X_train))**2
            res2[i,j] = sum(vec2[:,j])/train_size
            vec3[:,j] = (y_train-reg3.predict(X_train))**2
            res3[i,j] = sum(vec3[:,j])/train_size
        

    for j in range(J):
        for i in range(n):
            somme[j] = somme[j] + res[i,j]
            somme2[j] = somme2[j] + res2[i,j]
            somme3[j] = somme3[j] + res3[i,j]
    

    #plot the results
    plt.plot(axes,somme/n, label='OMP')
    plt.plot(axes,somme2/n, label='Lasso')
    plt.plot(axes,somme3/n,label='Lars')
    
    plt.xlabel('sparsity')
    plt.ylabel('train error')
    plt.title('Performance comparison on simulation data')
    plt.legend()


################################
#Average test squared error
################################

def test_error_data(n, J, x, y, test_size, train_size):
    #initialisation : 3 methods (Lasso, Lars, OMP)
    vec = np.zeros(test_size*J).reshape(test_size,J)
    res = np.zeros(n*J).reshape(n,J)    
    somme = np.zeros(J)
    vec2 = np.zeros(test_size*J).reshape(test_size,J)
    res2 = np.zeros(n*J).reshape(n,J)
    somme2 = np.zeros(J)
    vec3 = np.zeros(test_size*J).reshape(test_size,J)
    res3 = np.zeros(n*J).reshape(n,J)
    somme3 = np.zeros(J)
    axes = np.arange(1,11)

    #average test sqaured error : n iterations and sparsity (1 to J)
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(x, y,train_size=train_size)
        for j in range(J):
            reg2 = Lasso(alpha=1/(j+1)).fit(X_train, y_train)
            reg = OrthogonalMatchingPursuit(n_nonzero_coefs=j+1).fit(X_train, y_train)
            reg3 = Lars(n_nonzero_coefs=j+1).fit(X_train, y_train)
            vec[:,j] = (y_test-reg.predict(X_test))**2
            res[i,j] = sum(vec[:,j])/test_size
            vec2[:,j] = (y_test-reg2.predict(X_test))**2
            res2[i,j] = sum(vec2[:,j])/test_size
            vec3[:,j] = (y_test-reg3.predict(X_test))**2
            res3[i,j] = sum(vec3[:,j])/test_size
        

    for j in range(J):
        for i in range(n):
            somme[j] = somme[j] + res[i,j]
            somme2[j] = somme2[j] + res2[i,j]
            somme3[j] = somme3[j] + res3[i,j]
    

    #plot the results
    plt.plot(axes,somme/n, label='OMP')
    plt.plot(axes,somme2/n, label='Lasso')
    plt.plot(axes,somme3/n,label='Lars')
    
    plt.xlabel('sparsity')
    plt.ylabel('test error')
    plt.title('Performance comparison on simulation data')
    plt.legend()
  