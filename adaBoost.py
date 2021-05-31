import math
import numpy as np

# NOTE you can use if for Q1 c in tutorial9

def run_adaBooster_setup():
    """ Ask the user to enter the needed parameters
    """  
    print("Please insert the dataset. Each sample should be divided by a space and each coordinate within a sample should be divided by a coma. \nBe careful not to enter spaces after the coma that separates the samples' coordinates.\n")
    print("I.E.: x1=[1,2], x2=[-3,4], x3=[5,3] would be ---> 1,2 -3,4 5,3 \n")
    dataset_input = str(input())
    dataset = np.array([]).reshape(0,2)
    samples = dataset_input.split(' ')
    for sample in samples:
        coordinates = sample.split(',')
        coordinates = np.array([float(c) for c in coordinates])
        dataset = np.concatenate((dataset, [coordinates]))
    print("\n")
    print("Please insert the labels (y) of each sample in the dataset. \nYou should provide a sequential list of y where each y is separated by a single space.")
    print("I.E. +1 -1 +1 -1")
    labels = [ int(o) for o in str(input()).split(' ')] 
    print("\n")
    print("Please insert the number of weak classifiers:")
    n_classifiers = int(input())
    print("\n")
    thresholds = np.array([])
    for i in range(0, n_classifiers):
        print(f"Please insert the decision threshold for weak classifier {i+1} so that it classify a sample to be +1:")
        print("I.E.: x1 > 0")
        print("I.E.: x2 > 3")
        print("I.E.: x1 >= -4")
        thresholds = np.concatenate((thresholds, [str(input())]))
        print('\n')
    print("Please enter the target training error (when the adaboost should terminate). This should be normalised to the total number of samples:")
    print("I.E.: 0 -> if you want to stop it when the classifier classifies correctly all the training samples.")
    print("I.E.: 0.25 -> if you want to stop it when the classifier classifies correctly 75\% of the samples.")
    target_error = int(input())
    print("\n")
    print("Please enter the maximum number of iterations you want the algorithm to run for:")
    max_iterations = int(input())
    print("\n")
    return dataset, labels, n_classifiers, thresholds, target_error, max_iterations


# Decision stump used as weak classifier
class DecisionStump():
    def __init__(self, id, threshold=None):
        self.threshold = threshold
        self.id = id
        

    def predict(self, sample):
        if '>' in self.threshold:
            terms = self.threshold.split(' ')
            axis_in_condition = [int(c) - 1 for c in terms[0] if c.isdigit()]
            if sample[axis_in_condition[0]] > float(terms[2]):
                return 1
            else:
                return -1
        elif '<' in self.threshold:
            terms = self.threshold.split(' ')
            axis_in_condition = [int(c) - 1 for c in terms[0] if c.isdigit()]
            if sample[axis_in_condition[0]] < float(terms[2]):
                return 1
            else:
                return -1
        elif '<=' in self.threshold:
            terms = self.threshold.split(' ')
            axis_in_condition = [int(c) - 1 for c in terms[0] if c.isdigit()]
            if sample[axis_in_condition[0]] <= float(terms[2]):
                return 1
            else:
                return -1
        elif '>=' in self.threshold:
            terms = self.threshold.split(' ')
            axis_in_condition = [int(c) - 1 for c in terms[0] if c.isdigit()]
            if sample[axis_in_condition[0]] >= float(terms[2]):
                return 1
            else:
                return -1


class Adaboost():

    def __init__(self, n_clf, thresholds, target_error, max_iterations=10):
        self.n_clf = n_clf
        self.clfs = np.array([])
        for i in range (0, len(thresholds)):
            self.clfs = np.concatenate(( self.clfs, [DecisionStump(i+1, thresholds[i])] ))
        # self.alpha = 0
        self.alpha = []
        self.target_error = target_error
        self.max_iterations = max_iterations
        self.best_classifiers = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))

        iteration = 1
        while True:
            print(f"Iteration {iteration}: \nWeights: {w}")

            # Iterate through classifiers and find the best one
            lowest_error = 100000000
            best_classifier = 0
            for i in range(0, self.n_clf):
                clf = self.clfs[i]
                # Calculate Error
                err = 0
                for j, sample in enumerate(X):
                    prediction = clf.predict(sample)
                    error = 0 if prediction == y[j] else 1
                    err += error * w[j]
                if err < lowest_error:
                    lowest_error = err
                    best_classifier = i

            self.best_classifiers.append(self.clfs[best_classifier])
            print(f"Best classifier: {best_classifier+1}")

            # Get predictions from best classifer for each sample
            predictions = []
            for j, sample in enumerate(X):
                prediction = self.clfs[best_classifier].predict(sample)
                predictions.append(prediction)

            # Calculate weighted training error of best classifier
            weighted_error = 0
            for i, prediction in enumerate(predictions):
                # error += 0 if prediction == y[i] else 1
                w_e = 0 if prediction == y[i] else 1
                w_e *= w[i]
                weighted_error += w_e
            error /= len(y)
            print(f"Best classifier's weighted training error: {weighted_error}")
            
            
            # Calculate Alpha
            EPS = 1e-10
            alpha = 0.5 * np.log((1.0 - lowest_error) / (lowest_error))
            self.alpha.append(alpha)
            print(f"Alpha: {alpha}")

            # Calculate weights for next iteration
            new_w = []
            for i, weight in enumerate(w):
                new_w.append(w[i] * (np.exp(- alpha * y[i] * predictions[i])))
                print(f"Update weight: W{iteration}(sample{i+1})*e^-alpha{iteration}*y{i+1}*h{iteration}(sample{i+1}) ----> {w[i] * (np.exp(- alpha * y[i] * predictions[i]))}")
            # Normalize to one
            Z_normalisation = 0
            for i, weight in enumerate(new_w):
                Z_normalisation += weight
            for i, weight in enumerate(new_w):
                new_w[i] /= Z_normalisation
            print(f"Normalisation Z{iteration} when updating new weights: {Z_normalisation}")
            # Update weights for next iteration
            w = new_w



            # Check if the classifier has reached the desired target error
            # Find the output*alpha of each classifier for each sample
            tot_error = 0
            decision_formula = ''
            sample_classifications = np.zeros((X.shape[0], len(self.alpha)))
            for i, alpha in enumerate(self.alpha):  
                clf = self.best_classifiers[i]
                for j, sample in enumerate(X):
                    prediction = clf.predict(sample)
                    sample_classifications[j][i] = alpha if prediction == y[j] else -alpha
                decision_formula += f"{alpha} * h{clf.id}(x) + "
            # Calculate the AdaBooster classification error in this round
            sample_classifications = sample_classifications.sum(axis=1)
            for i, classification in enumerate(sample_classifications):
                classification = 1 if classification >= 0 else -1
                tot_error += 1/X.shape[0] if classification == y[j] else 0
            print(f"AdaBoost Classifier in this round: {decision_formula[:-2]}")
            print(f"AdaBoost Classifier (unweighted) error in this round: {tot_error}")
            # If the error is below our target error stop the execution
            if tot_error <= self.target_error:
                print('\n')
                print(f"The final hard classifier is: sgn({decision_formula[:-2]})")
                return

            # If we have reached the max iterations stop the execution
            if iteration >= self.max_iterations:
                return
            iteration += 1
            print("\n")



    def predict(self, sample):
        tot_error = 0
        sample_classifications = np.zeros((1, len(self.alpha)))
        for i, alpha in enumerate(self.alpha):  
            clf = self.best_classifiers[i]
            prediction = clf.predict(sample)
            sample_classifications[0][i] = alpha if prediction == 1 else -alpha
        # Calculate the AdaBooster classification error in this round
        sample_classifications = sample_classifications.sum(axis=1)
        return 1 if sample_classifications[0] >= 0 else -1

        
            
            

            




if __name__ == '__main__':
    dataset, labels, n_classifiers, thresholds, target_error, max_iterations = run_adaBooster_setup()
    classifier = Adaboost(n_classifiers, thresholds, target_error, max_iterations)
    classifier.fit(dataset, labels)
    
    while True:
        print("\nEnter a new sample (separating its coordinates with a coma and not including spaces) to predict or simply enter quit()")
        sample = str(input())
        if sample == 'quit()':
            quit()
        result = classifier.predict([float(c) for c in sample.split(',')])
        print(f"The AdaBoost classifier classified it as {result}")

        
