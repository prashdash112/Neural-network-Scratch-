import numpy as np
import math

# Activation functions for artificial neural networks

def identity(x):

    return x

def binary_step(x):

    if x<0:
        return 0
    else:
        return 1
    
def logistic(x):
    
    exp=1/(1+np.exp(-x))
    return exp

def tanh(x):
    
    exp=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return exp

def relu(x):

    if x<=0:
        return 0
    else:
        return x
    
def gelu(x):

    exp= x*(1+ math.erf(x/math.sqrt(2)))/2
    return exp

def softplus(x):

    exp=math.log(1+ np.exp(x))
    return exp

def elu(alpha,x):

    if x<=0:
        return alpha*(np.exp(x)-1)
    else:
        return x
    
def selu(x):

    alpha=1.67326
    fact=1.0507

    if x<0:
        return fact*alpha*(np.exp(x)-1)
    else:
        return fact*x
    
def prelu(alpha,x):

    if x<0:
        return alpha*x
    else:
        return x
    
def arctan(x):

    exp=np.arctan(x)
    return exp

def soft_sig(x):

    exp= x/(1+ abs(x))
    return exp

def sqnl(x):

    if x>2.0:
        return 1
    elif 0 <= x <=2.0:
        exp=(x-(x**2/4))
        return exp
    elif -2.0<= x <0:
        exp=(x+(x**2/4))
        return exp
    elif x<-2.0:
        return -1

def bent_identity(x):

    exp=((np.sqrt(x**2+1)-1)/2)+x
    return exp

def silu(x):

    exp=x/(1+np.exp(-x))
    return exp

def gaussian(x):

    return np.exp(-x**2)

def sq_rbf(x):

    if abs(x)<=1:
        return 1-(x**2/2)
    elif 1<=abs(x)<=2:
        return ((2-abs(x)**2)/2)
    elif abs(x)>=2:
        return 0