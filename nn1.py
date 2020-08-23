import numpy as np

def sigmoid(k): # activation function
    exp= 1.0/(1+ np.exp(-k))
    return exp
    
def sigmoid_derivative(r): #sigmoid derivative
    return r * (1.0 - r)


class neuralnet:
    
    def __init__(self,x,y):
        
        self.input=x
        self.y=y
        self.weight1=np.random.rand(self.input.shape[1],4)
        self.weight2=np.random.rand(4,1)
        self.output=np.zeros(y.shape)
        
    def feedforward(self):
        
        self.layer1=sigmoid(np.dot(self.input,self.weight1))
        self.output=sigmoid(np.dot(self.layer1,self.weight2))
        
    def backpropogation(self):
        
        b_weight2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        b_weight1 =np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weight2.T) * sigmoid_derivative(self.layer1))) 
        #np.dot(self.input.T, (2*(self.y - self.output) * sigmoid_derivative(self.output))) 

        # updating the weights with the derivative (slope) of the loss function
        self.weight1 += b_weight1
        self.weight2 += b_weight2
        
        

if __name__ == "__main__":
    
    X = np.array([[0,0,1],           # Training i/p
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    
    y = np.array([[0],[1],[1],[0]])  # Training o/p
    
    nn = neuralnet(X,y)

    for i in range(2500):    # Training the model for 2500 times
        
        nn.feedforward()     #Calculating the predicted output ŷ, known as feedforward
        nn.backpropogation() #Updating the weights and biases, known as backpropagation

    print(nn.output)         #it will print the predicted output for the given data using cost function optimization