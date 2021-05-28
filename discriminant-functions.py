# Discriminant Functions [week 2]
import numpy as np

def sequential_multiclass_perceptron_learning (N, augmented_matrix, eta, omega, number_of_classes ):
  # N is the number of exemplars provided in the question
  # augmented_matrix is the augmented feature vector from the question
  # eta is the learning rate given in the question
  # omega is an array containing all the output classes of the feature vectors

  N_counter = 0 # counter to check for convergence

  #Step 2. Initialise aj for each class
  at = np.zeros((number_of_classes, number_of_classes))

  for i in range(0,15):
    print ('Iteration: ',i+1)
    # Step 3. Find values of g1, g2 and g3 and then select the arg max of g
    index = i % 3

    #Print updated a^t value
    print('a^t:')
    print(at)
    
    # Compute g value
    #print('at is ', at)
    #print('aug matrix is',augmented_matrix[:,index])
    # g1 = at[0] @ augmented_matrix[:,index]
    # g2 = at[1] @ augmented_matrix[:,index]
    # g3 = at[2] @ augmented_matrix[:,index]

    g = np.empty([number_of_classes])
    for i in range(len(g)):
      g[i] = at[i] @ augmented_matrix[:,index]


    print('g1 | g2 | g3')
    print(g)

    #Step 4. Select the winner
    #Logic for 0,0,0 case and similar ones where 2 gs can produce max value
    seen = []
    bRepeated = False
    # Check if there are multiple max values, and assign the winner class accordingly
    for number in g:
        if number in seen:
          bRepeated = True
          print ("Number repeated!")
          m = max(g)
          temp = [index for index, j in enumerate(g) if j == m]
          winner_class = max(temp) + 1
        else:
            seen.append(number)
    #If all g values are unique, simply select the max value's class as the winner
    if(bRepeated == False):
      g = g.tolist()
      arg_max = max(g)
      winner_class = g.index(arg_max) + 1
    
    print('Winner class = ', winner_class, ', and actual class is:',omega[index])

    #Compare winnner to actual class 
    if(winner_class != omega[index]):
      # Step 4. Apply the update rule as per the algorithm 
      
      #Increment the actual class value which is incorrectly classified 
      at[omega[index]-1] = at[omega[index]-1] + eta * augmented_matrix[:,index]
      print('New loser value:', at[omega[index]-1])

      #Penalize the wrongly predicted Winner class
      at[winner_class-1] = at[winner_class-1] - eta * augmented_matrix[:,index]
      print('New winner value:', at[winner_class-1])

      #Reset counter to 0
      N_counter =0
    else:
      print ('No update is performed!')
      N_counter +=1
      if(N_counter == N +2):
        print('Learning has converged, so stopping...')
        print ('Final values of a^t after update....')
        print('at')
        print(at)
        break
      print ('N counter value = ', N_counter)
    print('at')
    print(at)
    print ('=========================================================')

if __name__ == "__main__":
    #Set input variables for the sequential_multiclass_perceptron_learning function
    N = 5
    eta = 1
    augmented_matrix = np.array([[1,1,1,1,1],[1,2,0,-1,-1],[1,0,2,1,-1]]) # Input matrix from the question
    omega = np.array([1,1,2,2,3]) # Class labels from the question 
    number_of_classes = 3

    #Call function
    sequential_multiclass_perceptron_learning (N, augmented_matrix, eta, omega, number_of_classes)