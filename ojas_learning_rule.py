import numpy as np
import pandas as pd

# Oja's Learning rule 
# through Mizan 
# this code works out zero-mean data so please enter the raw feature vectors as given in the question
# Tutorial 7, question 7

## -----------------------------------------------------------------------
# ONLY CHANGE THESE 3 INPUTS AND CHANGE THE NUMBER OF EPOCH YOU WANT WHEN APPLYING THE FUNCTION(Oja_learning_rule)
input_from_question = np.array([[[0,1]],[[3,5]],[[5,4]],[[5,6]],[[8,7]],[[9,7]]])
weight_x = np.array([[-1,0]])
learning_rate = 0.01
epochs = 6
## -----------------------------------------------------------------------

# we are going to perform Oja's learning on the input_vectors
input_vectors = []

mean_of_data = input_from_question.mean(axis=0)
for i in input_from_question:
    zero_mean_data = i - mean_of_data
    input_vectors.append(zero_mean_data)
    

def Oja_learning_rule(epoch):
    weight_update = np.copy(weight_x)  
    for i in range(1,epoch+1):
        df = pd.DataFrame({"x": [i for i in input_vectors]})
        df['y'] = df['x'].apply(lambda x: np.dot(x,weight_update.T))
        df['x - yw'] = df['x'].apply(lambda x: np.round(x, 4)) - df['y'].apply(lambda y: y * (weight_update))  
        df['ny(x -yw)'] = learning_rate * df['y'].apply(lambda y: y) * df['x - yw'].apply(lambda x: x)
        #Rounding the numbers         
        df['y'] = df['y'].apply(lambda y: np.round(y,4))
        df['x - yw'] =  df['x - yw'].apply(lambda x: np.round(x,4))
        df['ny(x -yw)']  = df['ny(x -yw)'].apply(lambda x: np.round(x,4))
        sum_of_weights = df['ny(x -yw)'].sum()
        weight_update = weight_update + sum_of_weights   
        display(df)
        print(f'after {i} epoch Total weight change is: {sum_of_weights}')
        print(f'after {i} epoch our weights are: {weight_update}')


Oja_learning_rule(epochs)