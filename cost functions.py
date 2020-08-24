#*COST FUNCTIONS FOR NEURAL NETWORK*

import numpy as np 
import math
import sys

math.log(sys.float_info.min * sys.float_info.epsilon)

# Sample values 
# True values 
Y_true = [0.17,4.23,11.46,23.87,25.39] 

# Predicted values 
Y_pred = [0.6,5.29,10.99,19.69,25.254]

# Mean Squared Error 
def MSE(Y_true,Y_pred):
    exp = np.square(np.subtract(Y_true,Y_pred)).mean()  # ytrue & ypred are arrays containing actual and predicted values 
    return exp

# Exponent cost function
def exponent_cost(Y_true,Y_pred,t):
    exp=t*np.exp( (1/t) * (np.sum(np.subtract(Y_true,Y_pred))**2 ))
    return exp
    
# Hellinger distance
def hellinger_distance(Y_true,Y_pred):
    exp=(1/np.sqrt(2))*np.sum(np.subtract(np.sqrt(Y_true),np.sqrt(Y_pred)) )**2
    return exp

# Itakura-Saito Distance
def ita_sa_distance(Y_true,Y_pred):
    exp=(np.subtract( (np.divide(Y_true,Y_pred)),(math.log(np.divide(Y_true,Y_pred))), ([1 for i in range(0,5)]) ))
    return exp
