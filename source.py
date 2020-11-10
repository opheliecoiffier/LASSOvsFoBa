#packages used
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit, Lars, lasso_path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


############################
#Alpha coefficients : Lasso
#############################
def alpha(X_train, train_size, nb_features, my_alphas):
    '''

    Parameters
    ----------
    X_train : data training points.
    train_size : number of training points.
    nb_features : number of features.
    my_alphas : array of different values for alpha.

    Returns : array of different values for alpha for Lasso method : 
              opposite order of my_alphas array.
    -------
    None.

    '''
    alpha_for_path, coefs_lasso, _ = lasso_path(X_train[:,0:nb_features-1],
                                                X_train[:,nb_features-1],alphas=my_alphas)
    return(alpha_for_path)


################################
#Average train squared error
################################

def train_error_data(n, J, x, y, train_size, nb_features, my_alphas):
    '''
    
    Parameters
    ----------
    n : number of repetitions.
    J : number of sparsity.
    x : data.
    y : desired output.
    train_size : number of training points.
    nb_features : number of features.
    my_alphas : array of different values for alpha.

    Returns : representation of MSE depending on sparsity for Lasso, OMP and Lars methods,
              for training points.
    -------

    '''
    #initialisation
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
            alpha_coef = alpha(X_train, train_size=train_size, nb_features=nb_features, 
                                     my_alphas=my_alphas)
            reg2 = Lasso(alpha=alpha_coef[j]).fit(X_train, y_train)
            reg = OrthogonalMatchingPursuit(n_nonzero_coefs=j+1).fit(X_train, y_train)
            reg3 = Lars(n_nonzero_coefs=j+1).fit(X_train, y_train)
            vec[:,j] = (y_train-reg.predict(X_train))**2
            res[i,j] = sum(vec[:,j])/train_size
            vec2[:,j] = (y_train-(reg2.predict(X_train)))**2
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

def test_error_data(n, J, x, y, test_size, train_size, nb_features, my_alphas):
    '''

    Parameters
    ----------
    n : number of repetitions.
    J : number of sparsity.
    x : data.
    y : desired output.
    test_size : number of test points.
    train_size : number of training points.
    nb_features : number of features.
    my_alphas : array of different values for alpha.

    Returns : representation of MSE depending on sparsity for Lasso, OMP and Lars methods,
              for test points.
    -------

    '''
    #initialisation 
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
            alpha_coef = alpha(X_train, train_size=train_size, nb_features=nb_features, 
                                     my_alphas=my_alphas)
            reg2 = Lasso(alpha=alpha_coef[j]).fit(X_train, y_train)
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
  
    
def alpha_choice_fig(x, y, my_alphas, nb_features, train_size):
    '''

    Parameters
    ----------
    x : data.
    y : desired output.
    my_alphas : array of different values for alpha.
    nb_features : number of features.
    train_size : number of train points.

    Returns : representation of lasso path
    -------

    '''
    X_train, X_test, y_train, y_test = train_test_split(x, y,train_size=train_size)
    alpha_for_path, coefs_lasso, _ = lasso_path(X_train[:,0:nb_features-1],
                                                    X_train[:,nb_features-1],alphas=my_alphas)
    for i in range(coefs_lasso.shape[0]):
        plt.plot(alpha_for_path,coefs_lasso[i,:])
        
    plt.xlabel('Alpha')
    plt.ylabel('Coefficients')
    plt.title('Lasso path')
    plt.show()


def alpha_choice(x, y, my_alphas, nb_features, train_size):
    '''

    Parameters
    ----------

    x : data.
    y : desired output.
    my_alphas : array of different values for alpha.
    nb_features : number of features.
    train_size : number of train points.

    Returns : array of non_zero coefficients depending on alpha
    ------

    '''
    X_train, X_test, y_train, y_test = train_test_split(x, y,train_size=train_size)
    alpha_for_path, coefs_lasso, _ = lasso_path(X_train[:,0:nb_features-1],
                                                X_train[:,nb_features-1],alphas=my_alphas)   
    nbNonZero = np.apply_along_axis(func1d=np.count_nonzero,arr=coefs_lasso,axis=0)
    tab = pd.DataFrame({'alpha':alpha_for_path,'Nb non-zero coefs':nbNonZero})
    return(tab)
