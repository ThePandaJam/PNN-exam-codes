import numpy as np
import pandas as pd

# Oja's Learning rule 
# through Mizan 
# this code works out zero-mean data so please enter the raw feature vectors as given in the question
# Tutorial 7, question 7

## -----------------------------------------------------------------------
# ONLY CHANGE THESE 4 INPUTS BELOW
input_from_question = np.array([[0,1],[3,5],[5,4],[5,6],[8,7],[9,7]])
weight_x = np.array([[-1,0]])
learning_rate = 0.01
epochs = 2
## -----------------------------------------------------------------------

# we are going to perform Oja's learning on the zero mean data below
zero_mean_data1 = []

mean_of_data1 = input_from_question.mean(axis=0)
for i in input_from_question:
#     x = np.array([i])
    sample_minus_mean = i - mean_of_data1
    zero_mean_data1.append(sample_minus_mean)

def Oja_learning_rule(epoch):
    weight_update = np.copy(weight_x)  
    for i in range(1,epoch+1):
        df = pd.DataFrame({"x": [i for i in zero_mean_data1]})
        df['y'] = df['x'].apply(lambda x: np.dot(x,weight_update.T))
        df['x - yw'] = df['x'].apply(lambda x: np.round(x, 6)) - df['y'].apply(lambda y: y * (weight_update))  
        df['ny(x -yw)'] = learning_rate * df['y'].apply(lambda y: y) * df['x - yw'].apply(lambda x: x)
        #Rounding the numbers         
        df['y'] = df['y'].apply(lambda y: np.round(y,6))
        df['x - yw'] =  df['x - yw'].apply(lambda x: np.round(x,6))
        df['ny(x -yw)']  = df['ny(x -yw)'].apply(lambda x: np.round(x,6))
        sum_of_weights = df['ny(x -yw)'].sum()
        weight_update = weight_update + sum_of_weights   
        display(df)
        print(f'after {i} epoch Total weight change is: {sum_of_weights}')
        print(f'after {i} epoch our weights are: {weight_update}')


print(f"Zero mean data:{zero_mean_data1}")
Oja_learning_rule(epochs)