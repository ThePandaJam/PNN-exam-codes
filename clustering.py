import math
import numpy as np

def run_hierarchical_clustering_setup():
    """ Ask the user to enter the needed parameters for hierarchical clustering

    Returns:
        [int, np.array, string]: c -> the number of classes to cluster.
                             dataset -> The dataset to cluster.
                             similarity_method -> the distancing method to use.
    """    
    print("Please enter the number of clusters you want to divide the dataset in and then press ENTER: ")
    c = int(input())
    print("\n")
    print("Please insert the dataset. Each sample should be divided by a space and each coordinate within a sample should be divided by a coma. \n Be careful not to enter spaces after the coma that separates the samples' coordinates.")
    print("I.E.: x1=[1,2], x2=[-3,4], x3=[5,3] would be ---> 1,2 -3,4 5,3")
    dataset_input = str(input())
    print("\n")
    dataset = []
    samples = dataset_input.split(' ')
    for sample in samples:
        coordinates = sample.split(',')
        coordinates = [float(c) for c in coordinates]
        dataset.append(coordinates)
    print("Please type the number corresponding to the similarity method to use and press ENTER:")
    print("1) Single-link")
    print("2) Complete-link")
    print("3) Group-average")
    print("4) Centroid")
    similarity_method = int(input())
    print("\n")
    similarity_options = {1: "single-link", 2:"complete-link", 3:"group-average", 4: "centroid"}
    similarity_method = similarity_options.get(similarity_method)
    return c, dataset, similarity_method

def run_competitive_learning_setup():
    """ Ask the user to enter the needed parameters for competitive learning.

    Returns:
        [string, np.array, np.array, float, np.array]: 
                             mode -> Specify if using normalisation or not.
                             dataset -> The dataset to cluster.
                             clusters -> The coordinates of the initial centroids
                             lr -> The learning Rate 
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
    print("Please insert the initial centroids. Each centroid should be divided by a space and each coordinate within a centroid should be divided by a coma. \nBe careful not to enter spaces after the coma that separates the centroids' coordinates.\n")
    print("I.E.: c1=[1,2], c2=[-3,4], c3=[5,3] would be ---> 1,2 -3,4 5,3 \n")
    centroids_input = str(input())
    print("\n")
    clusters = np.array([]).reshape(0,2)
    centroids = centroids_input.split(' ')
    for centroid in centroids:
        coordinates = centroid.split(',')
        coordinates = np.array([float(c) for c in coordinates])
        clusters = np.concatenate((clusters, [coordinates]))
    print("Please insert the value of the Learning Rate and then press ENTER (make sure it is a float number - I.E. 0.1)")
    lr = float(input())
    print("\n")
    print("Please insert the order to follow when selecting the samples within the algorithm. \nYou should provide a sequential list of the indexes of the samples to select separated by single spaces.")
    print("I.E. 1 2 1 5 3")
    print("NOTE: indexes start at 1!")
    order_of_indexes = [ int(o) - 1 for o in str(input()).split(' ')]
    print ("\n")
    return mode, dataset, clusters, lr, order_of_indexes


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



class HierarchicalClustering():
    """ This class can be used to execute problems regarding Hierarchical Clustering
    """    
    def __init__(self, n_classes, dataset, similarity_method="single-link"):
        self.n = n_classes # Number of classes that need to be found
        self.dataset = dataset # Dataset of type np.array([[x11, x12], [x12, x22],...,[x1n, x2n]])
        self.similarity_method = similarity_method # either "single-link", "complete-link", "group-average", or "centroid"
        self.clusters = [np.array([s]) for s in self.dataset] # The initial clusters of type np.array([[x11, x12], [x12, x22],...,[x1n, x2n]])

    def run(self):
        """Run hierarchical clustering on the dataset to find self.n number of clusters.
        """        
        iteration = 1
        while(len(self.clusters) != self.n):
            closest_clusters = None
            closest_distance = None
            # Find the two closest clusters
            for i, cluster in enumerate(self.clusters):
                closest_cluster_to_i, distance_to_i = self.find_closest_cluster(i) #TODO
                if not closest_distance or distance_to_i < closest_distance:
                    closest_clusters = [i, closest_cluster_to_i]
                    closest_distance = distance_to_i
            
            # Merge closest_clusters
            merged_clusters = [self.clusters[closest_clusters[0]], self.clusters[closest_clusters[1]]]
            self.merge_clusters(closest_clusters[0], closest_clusters[1])
            self.print_iteration(iteration, merged_clusters, closest_distance)
            iteration += 1
        self.print_final()

    def find_closest_cluster(self, cluster_index):
        """Find the the cluster closest to self.clusters[cluster_index] according to a given similarity method.
        Args:
            cluster_index (int): The index of the cluster in self.cluster
        Returns:
            ((int, float)): A pair containing the index of the cluster closest to self.clusters[cluster_index] and its distance.
        """        
        closest_cluster_index = None
        closest_distance = None
        for i, cluster in enumerate(self.clusters):
            similarity_method_options = {"single-link": lambda: self.get_single_link_distance(self.clusters[cluster_index], self.clusters[i]),
                                            "complete-link": lambda: self.get_complete_link_distance(self.clusters[cluster_index], self.clusters[i]),
                                            "group-average": lambda: self.get_average_link_distance(self.clusters[cluster_index], self.clusters[i]),
                                            "centroid": lambda: self.get_centroid_distance(self.clusters[cluster_index], self.clusters[i])}
            if cluster_index != i:
                func = similarity_method_options.get(self.similarity_method, lambda: "Invalid")
                distance = func()
                if not closest_distance or distance < closest_distance:
                    closest_cluster_index = i
                    closest_distance = distance

        return closest_cluster_index, closest_distance


            
    def get_single_link_distance(self, cluster_a, cluster_b):
        minimum_distance = None
        for a in cluster_a:
            for b in cluster_b:
                dist = np.linalg.norm(a-b)
                if not minimum_distance or minimum_distance > dist:
                    minimum_distance = dist
        return minimum_distance 

    def get_complete_link_distance(self, cluster_a, cluster_b):
        maximum_distance = None
        for a in cluster_a:
            for b in cluster_b:
                dist = np.linalg.norm(a-b)
                if not maximum_distance or maximum_distance < dist:
                    maximum_distance = dist
        return maximum_distance 
    
    def get_average_link_distance(self, cluster_a, cluster_b):
        distances = np.array([])
        for a in cluster_a:
            for b in cluster_b:
                dist = np.linalg.norm(a-b)
                distances = np.append(distances, np.array([dist]))
        return np.average(distances) 

    def get_centroid_distance(self, cluster_a, cluster_b):
        centroid_a = np.average(cluster_a) 
        centroid_b = np.average(cluster_b) 
        dist = np.linalg.norm(centroid_a - centroid_b)
        return dist 
    
    def merge_clusters(self, cluster_index_a, cluster_index_b):
        merged_cluster = np.concatenate((self.clusters[cluster_index_a], self.clusters[cluster_index_b]))
        self.clusters = [self.clusters[i] for i in range(0, len(self.clusters)) if i != cluster_index_a and i != cluster_index_b]
        self.clusters.append(merged_cluster)

    def print_iteration(self, i, merged_clusters, distance):
        print(f"End of iteration {str(i)} " )
        print(f"Merged clusters {merged_clusters[0]} and {merged_clusters[1]}. Distance between the clusters was {distance}.")
        print(f"There are {str(len(self.clusters))} clusters:")
        for j, c in enumerate(self.clusters):
            print(f"Cluster {j} -> {self.clusters[j]}")
        print ("\n")
    
    def print_final(self):
        print("FINAL CLUSTERS:")
        for j, c in enumerate(self.clusters):
            print(f"Cluster {j} -> {self.clusters[j]}")




class CompetitiveLearning:
    """ This class can be used to execute problems regarding Competitive Learning
    """ 
    def __init__(self, mode, dataset, clusters, lr, order_of_samples):
        self.mode = mode # Indicates wether to use normalisation or not. Either "with norm" or "without norm"
        self.dataset = dataset # Dataset of type np.array([[x11, x12], [x12, x22],...,[x1n, x2n]])
        self.centroids = clusters # Initial clusters of type np.array([[x11, x12], [x12, x22],...,[x1n, x2n]])
        self.lr = lr # Learning rate. Must be a float > 0
        self.order_of_samples = order_of_samples # Indicates what order to select the samples in the algorithm. 
                                                # Of type np.array([int, int, int]) where each int is the corresponding index to the sample in self.dataset

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

    def run_with_normalisation(self):
        augmented_dataset = np.array([np.insert(sample, 0, 1) for sample in self.dataset])
        normalised_dataset = np.array([np.divide(sample, np.linalg.norm(sample)) for sample in augmented_dataset])
        augmented_centroids = np.array([np.insert(sample, 0, 1) for sample in self.centroids])

        print(f"The augmented dataset is :{augmented_dataset}")
        print(f"The normalised dataset is :{normalised_dataset}\n")
        for iteration, i in enumerate(self.order_of_samples):
            x = normalised_dataset[i]
            print(f"Iteration {iteration+1}:")
            print(f"Selected x{i+1} {self.dataset[i]} which normalised is --> {x}")

            net_inner_products = np.array([np.multiply(c.transpose(), x) for c in augmented_centroids])
            print(f"The inner products to each centroid with respect to x{i+1} are {net_inner_products}")
            j = np.argmax(np.sum(net_inner_products, axis=1))

            rhino_centroid = augmented_centroids[j]
            print(f"The selected centroid is c{j+1} {rhino_centroid}, with a net inner product to {x} of {net_inner_products[j]}")
            
            # Update Rhino Centroid
            rhino_centroid = np.add(rhino_centroid, np.multiply([self.lr], x))
            print(f"Updated Centroid c{j+1} with respect to x{i+1}: {rhino_centroid}")
            # Normalise Rhino Centroid
            rhino_centroid = np.divide(rhino_centroid, np.linalg.norm(rhino_centroid))
            print(f"Normalised Centroid c{j+1} with respect to x{i+1}: {rhino_centroid} \n")
            augmented_centroids[j] = rhino_centroid

        self.centroids = augmented_centroids
        print(f"The final Centroids are: {self.centroids}\n")
    
    def run_without_normalisation(self):
        for iteration, i in enumerate(self.order_of_samples):
            print(f"Iteration {iteration+1}:")
            x = self.dataset[i]

            distances_to_centroids = np.array([np.linalg.norm(x - c) for c in self.centroids])
            j = distances_to_centroids.argmin()

            rhino_centroid = self.centroids[j]
            print(f"The selected centroid is c{j+1} {rhino_centroid}, with a distance of {distances_to_centroids[j]} to {x}")
            
            # Update Rhino Centroid
            rhino_centroid = np.add(rhino_centroid, np.multiply([self.lr], np.subtract(x, rhino_centroid)))
            self.centroids[j] = rhino_centroid
            print(f"Updated Centroid with respect to x{i+1}: {rhino_centroid} \n")
        print(f"The final Centroids are: {self.centroids}\n")
    
    def classify_samples(self):
        """Prints what cluster each sample belongs to.
            NOTE Only works in 'without norm' mode
        """ 
        if self.mode == 'without norm':
            for j, sample in enumerate(self.dataset):
                minimum_distance = None
                closest_centroid_index = None
                for i, centroid in enumerate(self.centroids):
                    dist = np.linalg.norm(sample - centroid)
                    if not closest_centroid_index or minimum_distance > dist:
                        minimum_distance = dist
                        closest_centroid_index = i
                print(f"Sample x{j+1} {sample} belongs to cluster c{closest_centroid_index+1} {self.centroids[closest_centroid_index]}")
            print("\n")
        elif self.mode == 'with norm':
            print("This feature is not available in mode \"with normalisation\". No examples were given.")
            print("\n")

    
    def classify_new_data(self):
        """Asks the user to input a new sample to classify and says what cluster it belongs to.
            NOTE Only works in 'without norm' mode
        """        
        if self.mode == 'without norm':
            print("Please insert the new sample to classify. Do not insert spaces and divide its coordinates with a coma")
            new_sample = np.array([float(c) for c in str(input()).split(',')])
            print('\n')
            minimum_distance = None
            closest_centroid_index = None
            for i, centroid in enumerate(self.centroids):
                dist = np.linalg.norm(new_sample - centroid)
                if not closest_centroid_index or minimum_distance > dist:
                    minimum_distance = dist
                    closest_centroid_index = i
            print(f"The new sample {new_sample} belongs to cluster c{closest_centroid_index+1} {self.centroids[closest_centroid_index]}")
        elif self.mode == 'with norm':
            print("This feature is not available in mode \"with normalisation\". No examples were given.")
            print("\n")



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
                    if not closest_centroid_index or minimum_distance > dist:
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
                if not closest_centroid_index or minimum_distance > dist:
                    minimum_distance = dist
                    closest_centroid_index = i
            print(f"The new sample {new_sample} belongs to cluster c{closest_centroid_index+1} {self.centroids[closest_centroid_index]}")
        elif self.mode == 'with norm':
            print("This feature is not available in mode \"with normalisation\". No examples were given.")
            print("\n")



if __name__ == '__main__':

    print("Select what algorithm to run. Press its number and press ENTER.")
    print("1) Hierarchical Clustering")
    print("2) Competitive Algorithm")
    print("3) Leader Follower\n")

    algo = int(input())
    print("")

    if algo == 1:
        # Hierarchical Clustering Exercise.
        c, dataset, similarity_method = run_hierarchical_clustering_setup()
        cluster = HierarchicalClustering(c, dataset, similarity_method)
        cluster.run()
    elif algo == 2:
        # Competitive Algorithm
        mode, dataset, clusters, lr, order_of_indexes = run_competitive_learning_setup()
        cluster = CompetitiveLearning(mode, dataset, clusters, lr, order_of_indexes)
        cluster.run()
    elif algo == 3:
        # Leader follower algorithm
        mode, dataset, lr, theta, order_of_indexes = run_leader_follower_setup()
        cluster = LeaderFollower(mode, dataset, lr, theta, order_of_indexes)
        cluster.run()



