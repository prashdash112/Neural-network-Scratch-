import tensorflow as tf

class WideAndDeepModel(tf.keras.Model):
    def __init__(self,units=30,activation='relu',**kwargs ):
        super().__init__(**kwargs)
        self.hidden1=tf.keras.layers.Dense(units=units,activation=activation)
        self.hidden2=tf.keras.layers.Dense(units=(units/2),activation=activation)
        self.main_output = tf.keras.layers.Dense(1)
        self.aux_output = tf.keras.layers.Dense(1)
        
    def call(self,inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output
    
    
model = WideAndDeepModel()
