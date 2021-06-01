import math
import numpy as np

def run_leader_follower_setup():
    """ Ask the user to enter the needed parameters

    Returns:
        [string, np.array, float, float, np.array]: 
                             mode -> Specify if using normalisation or not.
                             dataset -> The dataset to cluster.
                             lr -> The learning Rate 
                             theta -> The threshold theta.
                             order_of_indexes -> The order to follow when selecting samples in the algorithm
    """    
    print("How you want to run the algorithm? (Enter the corresponding number and press ENTER)")
    print("1) With normalisation and argumentation")
    print("2) Without normalisation and argumentation")
    mode = int(input())
    algorithm_variants = {1: "with norm", 2:"without norm"}
    mode = algorithm_variants.get(mode)
    print("\n")
    print("Please insert the dataset. Each sample should be divided by a space and each coordinate within a sample should be divided by a coma. \nBe careful not to enter spaces after the coma that separates the samples' coordinates.\n")
    print("I.E.: x1=[1,2], x2=[-3,4], x3=[5,3] would be ---> 1,2 -3,4 5,3 \n")
    dataset_input = str(input())
    print("\n")
    dataset = np.array([]).reshape(0,2)
    samples = dataset_input.split(' ')
    for sample in samples:
        coordinates = sample.split(',')
        coordinates = np.array([float(c) for c in coordinates])
        dataset = np.concatenate((dataset, [coordinates]))
    print("Please insert the value of the learning rate and then press ENTER (make sure it is a float number - I.E. 0.1)")
    lr = float(input())
    print("\n")
    print("Please insert the value of the the threshold theta and then press ENTER (make sure it is a float number - I.E. 2.0)")
    theta = float(input())
    print("\n")
    print("Please insert the order to follow when selecting the samples within the algorithm. \nYou should provide a sequential list of the indexes of the samples to select separated by single spaces.")
    print("I.E. 1 2 1 5 3")
    print("NOTE: indexes start at 1!")
    order_of_indexes = [ int(o) - 1 for o in str(input()).split(' ')]
    print ("\n")
    return mode, dataset, lr, theta, order_of_indexes


class LeaderFollower:
    """ This class can be used to execute problems regarding Leader Follower
    """ 
    def __init__(self, mode, dataset, lr, theta, order_of_samples):
        self.mode = mode # Indicates wether to use normalisation or not. Either "with norm" or "without norm"
        self.dataset = dataset # Dataset of type np.array([[x11, x12], [x12, x22],...,[x1n, x2n]])
        self.lr = lr # Learning rate. Must be a float > 0
        self.theta = theta # The threshold to use in the algorithm.
        self.order_of_indexes = order_of_indexes # Indicates what order to select the samples in the algorithm. 
                                                # Of type np.array([int, int, int]) where each int is the corresponding index to the sample in self.dataset
        self.centroids = np.array([]) # Empty np.array of initial centroids

    
    def run(self):
        """Runs the algorithm on the samples listed in self.order_of_samples.
            After that, the user can select some more operations to do with the updated clusters.
        """        
        if self.mode == "with norm":
            self.run_with_normalisation()
        elif self.mode == "without norm":
            self.run_without_normalisation()
        user_input = 0
        while user_input != 3:
            # While the user doesnt selects exit show some options.
            # NOTE only applicable when in "without norm" mode.
            if user_input == 1:
                self.classify_samples()
            elif user_input == 2:
                self.classify_new_data()

            print("What do you want to do now?")
            print("1) Classify all the existing samples")
            print("2) Classify a new sample")
            print("3) Exit")
            user_input = int(input())

    def run_without_normalisation(self):
        # Shape the list of centroid to the right shape
        self.centroids = np.array([]).reshape(0, self.dataset.shape[1])
        # Initialise first centroid
        self.centroids = np.concatenate( (self.centroids, [self.dataset[self.order_of_indexes[0]]]) )
        for iteration, i in enumerate(self.order_of_indexes):
            x = self.dataset[i]
            print(f"Iteration {iteration}:")
            print(f"Selected x{i+1} {x}")

            distances_to_centroids = np.array([np.linalg.norm(x - c) for c in self.centroids])
            print(f"Distances to each centroids are {distances_to_centroids}")
            j = distances_to_centroids.argmin()
            rhino_centroid = self.centroids[j]
            print(f"The closest centroid is c{j+1} {rhino_centroid}")

            if np.linalg.norm(x - rhino_centroid) < self.theta:
                print(f"C{j+1} is within the threshold")
                self.centroids[j] = np.add( rhino_centroid, np.multiply(self.lr, np.subtract(x, rhino_centroid)) )
                print(f"Updated centroid c{j+1} to be {self.centroids[j]}\n")
            else:
                print(f"C{j+1} is not within the threshold")
                self.centroids = np.concatenate( (self.centroids, [x]) )
                print(f"Added new Centroid {x}\n")
        
        print(f"The final Centroids are: {self.centroids}\n")

    def run_with_normalisation(self):

        augmented_dataset = np.array([np.insert(sample, 0, 1) for sample in self.dataset])
        normalised_dataset = np.array([np.divide(sample, np.linalg.norm(sample)) for sample in augmented_dataset])

        print(f"The augmented dataset is :{augmented_dataset}")
        print(f"The normalised dataset is :{normalised_dataset}\n")

        # Shape the list of centroid to the right shape (it must be augmented)
        self.centroids = np.array([]).reshape(0, normalised_dataset.shape[1])

        # Initialise first centroid
        self.centroids = np.concatenate( (self.centroids, [normalised_dataset[self.order_of_indexes[0]]]) )

        for iteration, i in enumerate(self.order_of_indexes):
            x = normalised_dataset[i]
            print(f"Iteration {iteration + 1}:")
            print(f"Selected x{i+1} {self.dataset[i]} which normalised is --> {x}")

            net_inner_products = np.array([np.multiply(c.transpose(), x) for c in self.centroids])
            print(f"The inner products to each centroid with respect to x{i+1} are {net_inner_products}")
            j = np.argmax(np.sum(net_inner_products, axis=1))
            rhino_centroid = self.centroids[j]
            print(f"The closest centroid is c{j+1} {rhino_centroid}")

            if np.linalg.norm(x - rhino_centroid) < self.theta:
                print(f"C{j+1} is within the threshold, as {np.linalg.norm(x - rhino_centroid)} < {self.theta}")
                rhino_centroid = np.add(rhino_centroid, np.multiply([self.lr], x)) # Update cluster center
                print(f"Updated Centroid C{j+1} with respect to x{i+1}: {rhino_centroid}") 
                rhino_centroid = np.divide(rhino_centroid, np.linalg.norm(rhino_centroid)) # Normalise updated cluster center
                print(f"Normalised Centroid c{j+1} with respect to x{i+1}: {rhino_centroid} \n")
                self.centroids[j] = rhino_centroid # Actually update the new cetroid
            else:
                print(f"C{j+1} is not within the threshold, as {np.linalg.norm(x - rhino_centroid)} > {self.theta}")
                new_centroid = np.divide(x, np.linalg.norm(x)) # Create new centroid
                self.centroids = np.concatenate( (self.centroids, [new_centroid]) )
                print(f"Added new centroid c{len(self.centroids - 1)} {new_centroid}")
        
        print(f"The final Centroids are: {self.centroids}\n")

    def classify_samples(self):
        if self.mode == 'without norm':
            for j, sample in enumerate(self.dataset):
                minimum_distance = None
                closest_centroid_index = None
                for i, centroid in enumerate(self.centroids):
                    dist = np.linalg.norm(sample - centroid)
                    if not minimum_distance or minimum_distance > dist:
                        minimum_distance = dist
                        closest_centroid_index = i
                print(f"Sample x{j+1} {sample} belongs to cluster c{closest_centroid_index+1} {self.centroids[closest_centroid_index]}")
            print("\n")
        elif self.mode == 'with norm':
            print("This feature is not available in mode \"with normalisation\". No examples were given.")
            print("\n")

    
    def classify_new_data(self):
        if self.mode == 'without norm':
            print("Please insert the new sample to classify. Do not insert spaces and divide its coordinates with a coma")
            new_sample = np.array([float(c) for c in str(input()).split(',')])
            print('\n')
            minimum_distance = None
            closest_centroid_index = None
            for i, centroid in enumerate(self.centroids):
                dist = np.linalg.norm(new_sample - centroid)
                if not minimum_distance or minimum_distance > dist:
                    minimum_distance = dist
                    closest_centroid_index = i
            print(f"The new sample {new_sample} belongs to cluster c{closest_centroid_index+1} {self.centroids[closest_centroid_index]}")
        elif self.mode == 'with norm':
            print("This feature is not available in mode \"with normalisation\". No examples were given.")
            print("\n")


if __name__=='__main__':
    # Leader follower algorithm.  NOTE youll be asked to enter all the parameters when executing the script
    mode, dataset, lr, theta, order_of_indexes = run_leader_follower_setup()
    cluster = LeaderFollower(mode, dataset, lr, theta, order_of_indexes)
    cluster.run()
