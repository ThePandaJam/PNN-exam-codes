'''
    File name: deep-gen-neural-nets.py
    Author: Camil Abraham Hamanoiel Faure
            Anish Thekkan
    Date created: 26/05/2021
    Date last modified: 30/05/2021
    Python Version: 3.7
'''

import math as math
import sympy as sy

#
# NOTE: This script only works for k=1. Examples with different values
#       of k are not provided in the lecture or the tutorial.
#

# Change this variables according to the question
training_iterations = 1 # Number of training iterations.
x_real = [[1,2],[3,4]] # Real samples
x_fake = [[5,6],[7,8]] # Generated samples
    # Both x_real and x_fake must have the same number of 
    # samples for part b to work

theta = [0.1,0.2] # Initial parameters of discriminator

prob_real = 0.5 # Probability of real samples to be selected
prob_fake = 0.5 # Probability of generated samples to be selected
    # In Tutorial 7 Q.3 each sample has the same probability,
    # that is why the current value is 0.5 for both

learning_rate = 0.02 # Part b Q.3 says that learning rate is 0.02 but
                    # it can be changed to the desired value

# DO NOT change this variables
# ----------------------------
# Here, symbols are created for each variable so that
# the discriminator function can be differentiated and
# evaluated with different values
n = len(x_real[0]) 
x = [None] * n
t = [None] * n
for i in range(n):
    name = 'x'+str(i)
    x[i] = sy.Symbol(name)
    name = 't'+str(i)
    t[i] = sy.Symbol(name)
# ----------------------------

# Change this according to the problem's discriminator function
discriminator_function = 1/(1+ math.e**-(t[0]*x[0] - t[1]*x[1] - 2))
    # The discriminator function can be changed to any function needed.
    # 
    # Note that the variables MUST be expressed as t[i] and x[i] 
    # where:
    # 't' is a list containig variables for the parameters of Discriminator (Theta)
    # 'x' is a list containig variables for each attribute of each sample
    # 'i' is the number of the attribute and parameter used in that case
    # 
    # i.e. for the equation: "θ1*x1" 
    # then above it should be written: "t[0]*x[0]"
    # because python lists start in 0

# *****DO NOT CHANGE CODE BELOW THIS LINE********
#
m = len(x_real)
m_fake = len(x_fake)

#
# Because it is discrete we don't need to find integrals here when computing expectations
#
print('************')
print('*****GAN****')
print('************')

#
# Tutorial 7 Question 3 part a
#
for iter in range(training_iterations):
    print ('\n--Start of',str(iter+1)+"°",'training iteration.--')
    V_D = 0 # Discriminator Value 

    # This for is to obtain the discriminator value
    for i in range(m):
        xx = x_real[i]
        res_discr_funct = discriminator_function
        for j in range(n):
            # .subs() is a function that replaces the x[i] and t[i] variables of the 
            # discriminant function with the actual values of the theta and real samples
            # given in the problem
            res_discr_funct = res_discr_funct.subs(x[j],xx[j]).subs(t[j],theta[j])
        V_D += prob_real*math.log(res_discr_funct)


    print('\n***** PART A ****\n')

    print('\nThe Discriminator value is', V_D)

    V_G = 0 # Generator Value

    # This for is to obtain the generator value
    for i in range(m_fake):
        xx = x_fake[i]
        res_discr_funct = discriminator_function
        for j in range(n):
            # .subs() is a function that replaces the x[i] and t[i] variables of the 
            # discriminant function with the actual values of the theta and real samples
            # given in the problem
            res_discr_funct = res_discr_funct.subs(x[j],xx[j]).subs(t[j],theta[j])
        V_G += prob_fake*math.log(1 - res_discr_funct)
        
    print('\nThe Generator value is', V_G)

    V_DG = V_D + V_G

    print('\nThe Computed V_DG is ', V_DG)

    #
    # Tutorial 7 Question 3 part b
    #

    print('\n\n*********************')
    print('***** PART B ****\n')

    alpha_beta = [[0] * n] * m
    for i in range(m):
        xx = x_real[i]
        xx_bar = x_fake[i]
        # Two different equations are needed. One with x_real[i] and another with
        # x_fake[i]. This is explained in the tutorial solutions.
        discriminator_function_xx = discriminator_function
        discriminator_function_xx_bar = discriminator_function
        for j in range(n):
            discriminator_function_xx = discriminator_function_xx.subs(x[j],xx[j])
            discriminator_function_xx_bar = discriminator_function_xx_bar.subs(x[j],xx_bar[j])

        learning_equation = sy.log(discriminator_function_xx) 
        learning_equation += sy.log(1 - discriminator_function_xx_bar)
        
        differential_equation = [None] * n
        # The learning equation is differentiated in this for
        for j in range(n):
            differential_equation[j] = learning_equation.diff(t[j])
            # After differentiating the learning equation, the next for replaces
            # thetha variables with the actual values to obtain the result
            for k in range(n):
                differential_equation[j] = differential_equation[j].subs(t[k],theta[k])
        # The result of the differentiated equations is added to a list
        alpha_beta[i] = differential_equation

    # Alpha and beta are added together to obtain delta
    delta_theta = [0] * n
    for i in range(n):
        delta_theta[i] = 0
        for j in range(m):
            print( 'Alpha and Beta', i+1,j+1, 'is', alpha_beta[j][i])
            delta_theta[i] += (1/m)*alpha_beta[j][i]

    print( '\nDelta is ', delta_theta,'\n')

    # Lastly, delta is added to the current theta values to obtain the new
    # theta values after training
    for i in range(n):
        theta[i] = theta[i] + learning_rate*delta_theta[i]
        print('New theta',i+1   ,'is',theta[i])
    print ('\n--End of',str(iter+1)+"°",'training iteration.--')
print ('\n\n---------End of the script.---------')