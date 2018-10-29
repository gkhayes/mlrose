""" Classes for defining bitstring optimization problems.

    Author: Genevieve Hayes <ghayes17@gmail.com>
    License: 3-clause BSD license.
"""

import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_tree

class BitString: 
    """Base class for bitstring optimisation problems."""
    
    def __init__(self, length, fitness_fn):  
        """Initialize BitString object.
    
        Args:
        length: int. Length of bitstring to be used in problem
        fitness_fn: fitness function object. Object to implement fitness function 
        for optimization.
           
        Returns:
        None
        """
        self.length = length
        self.state = np.array([0]*self.length)
        self.neighbors = []
        self.fitness_fn = fitness_fn
        self.fitness = 0
        
    def reset(self):
        """Set the current bitstring to a random value and get its fitness
        
        Args:
        None
           
        Returns:
        None
        """
        self.state = np.random.randint(0, 2, self.length)
        self.fitness = self.calculate_fitness(self.state)
    
    def get_length(self):
        """ Return the bitstring length.
        
        Args: 
        None
        
        Returns:
        length: int. Length of bitstring.
        """
        length = self.length
        return length
    
    def get_fitness(self):
        """ Return the fitness of the current bitstring.
        
        Args:
        None
           
        Returns:
        fitness: float. Fitness value of current bitstring.
        """
        fitness = self.fitness
        
        return fitness
        
    def get_state(self):
        """ Return the current bitstring
        
        Args:
        None
           
        Returns:
        state: array. Current bitstring value.
        """
        state = self.state
        
        return state
        
    def set_state(self, new_bitstring):
        """ Change the current bitstring to a specified value and get its fitness
        
        Args:
        new_bitstring: array. New bitstring value.
           
        Returns:
        None
        """
        self.state = new_bitstring
        self.fitness = self.calculate_fitness(self.state)        
    
    def calculate_fitness(self, bitstring):
        """Evaluate the fitness of a bitstring
        
        Args:
        bitstring: array. Bitstring value for evaluation.
           
        Returns:
        fitness: float. Value of fitness function. 
        """
        fitness = self.fitness_fn.evaluate(bitstring)
        
        return fitness
                 
    def find_neighbors(self):
        """Find all neighbors of the current bitstring
        
        Args:
        None
           
        Returns:
        None
        """
        self.neighbors = []
        
        for i in range(self.length):
            neighbor = np.copy(self.state)
            neighbor[i] = np.abs(neighbor[i] - 1)
            self.neighbors.append(neighbor)

    def best_neighbor(self):
        """Return the best neighbor of current bitstring
        
        Args:
        None
           
        Returns:
        best: array. Bitstring defining best neighbor.
        """
        fitness_list = []
        
        for n in self.neighbors:
            fitness = self.fitness_fn.evaluate(n)
            fitness_list.append(fitness)

        best = self.neighbors[np.argmax(fitness_list)]
        
        return best 
    
    def random_neighbor(self):
        """Return random neighbor of current bitstring
        
        Args:
        None
           
        Returns:
        neighbor: array. Bitstring of random neighbor.
        """
        neighbor = np.copy(self.state)
        i = np.random.randint(0, self.length)
        neighbor[i] = np.abs(neighbor[i] - 1)
        
        return neighbor

class Genetic(BitString):
    """Child class for solving bitstring optimisation problems using a genetic algorithm."""
    
    def __init__(self, length, fitness_fn):  
        """Initialize Genetic object.
    
        Args:
        length: int. Length of bitstring to be used in problem
        fitness_fn: fitness function object. Object to implement fitness function 
        for optimization.
           
        Returns:
        None
        """
        BitString.__init__(self, length, fitness_fn)
        self.population = []
        self.pop_fitness = []
        self.probs = []
    
    def random(self):
        """Return a random bitstring
        
        Args:
        None
           
        Returns:
        state: array. Randomly generated bitstring.
        """
        state = np.random.randint(0, 2, self.length)
        
        return state       
       
    def create_population(self, pop_size):
        """Create a population of random bitstrings
        
        Args:
        pop_size: int. Size of population to be created.
           
        Returns:
        None
        """
        population = []
        pop_fitness = []
        
        for i in range(pop_size):
            state = self.random()
            fitness = self.calculate_fitness(state)
            
            population.append(state)
            pop_fitness.append(fitness)
        
        self.population = np.array(population)
        self.pop_fitness = np.array(pop_fitness)
        
    def calculate_probs(self):
        """Calculate the probability of each member of the population reproducing.
        
        Args:
        None
           
        Returns:
        None
        """
        self.probs = self.pop_fitness/np.sum(self.pop_fitness)
    
    def get_population(self):
        """ Return the current population
        
        Args:
        None
           
        Returns:
        population: array. Numpy array containing current population.
        """
        population = self.population
        
        return population
    
    def set_population(self, new_population):
        """ Change the current population to a specified new population and get 
        the fitness of all members
        
        Args:
        new_population: array. Numpy array containing new population.
           
        Returns:
        None
        """
        self.population = new_population
        
        # Calculate fitness
        pop_fitness = []
        
        for i in range(len(self.population)):
            fitness = self.calculate_fitness(self.population[i])
            pop_fitness.append(fitness)
        
        self.pop_fitness = np.array(pop_fitness)
        
    def best_child(self):
        """Return the best bitstring in the current population.
        
        Args:
        None
           
        Returns:
        best: array. Bitstring defining best child.
        """
        best = self.population[np.argmax(self.pop_fitness)]
        
        return best

class ProbOpt(Genetic):
    """Child class for solving bitstring optimisation problems using a probabilistic
    optimization algorithm (i.e. MIMIC)."""
    
    def __init__(self, length, fitness_fn):  
        """Initialize ProbOpt object.
    
        Args:
        length: int. Length of bitstring to be used in problem
        fitness_fn: fitness function object. Object to implement fitness function 
        for optimization.
           
        Returns:
        None
        """
        Genetic.__init__(self, length, fitness_fn)
        self.keep_sample = []
        self.probs = np.zeros([2, self.length])
        self.parent = []

    def select_keep_sample(self, keep_pct):
        """Select samples with fitness in the top n percentile.
    
        Args:
        keep_pct: float. Proportion of samples to keep.
           
        Returns:
        None
        """
        # Determine threshold
        theta = np.percentile(self.pop_fitness, 100*(1 - keep_pct))

        # Determine samples for keeping
        keep_inds = np.where(self.pop_fitness >= theta)[0]
        
        # Determine sample for keeping
        self.keep_sample = self.population[keep_inds]
     
    def update_probs(self):
        """Update probability density estimates.
        
        Args:
        None
           
        Returns:
        None
        """
        # Create mutual info matrix
        mutual_info = np.zeros([self.length, self.length])
        for i in range(self.length -  1):
            for j in range(i + 1, self.length):
                mutual_info[i, j] = -1*mutual_info_score(self.keep_sample[:, i], \
                           self.keep_sample[:, j])
    
        # Find minimum spanning tree of mutual info matrix
        mst = minimum_spanning_tree(csr_matrix(mutual_info))
    
        # Convert minimum spanning tree to depth first tree with node 0 as root
        dft = depth_first_tree(csr_matrix(mst.toarray()), 0, directed=False)
        dft = dft.toarray()
        
        # Determine parent of each node
        parent = np.argmin(dft[:, 1:], axis = 0)
    
        # Get probs
        probs = np.zeros([2, self.length])
        
        if len(self.keep_sample) == 0:
            probs[:, 0] = 0
        else:
            probs[:, 0] = np.sum(self.keep_sample[:, 0])/len(self.keep_sample)
    
        for i in range(1, self.length):
            for j in range(2):
                subset = self.keep_sample[np.where(self.keep_sample[:, parent[i - 1]] == j)[0]]
                count = np.sum(subset[:, i])
                
                if len(subset) == 0:
                    probs[j, i] = 0
                else:
                    probs[j, i] = count/len(subset)
        
        # Update probs and parent
        self.probs = probs
        self.parent = parent
    
    def find_sample_order(self, parent):
        """Determine order in which to generate sample bits given parent node list
        
        Args:
        parent: list. List of nodes giving the parents of each node in bitstring tree
           
        Returns:
        sample_order: list. Order in which sample bits should be generated.
        """
        sample_order = []
        last = [0]
        parent = np.array(parent)
    
        while len(sample_order) < len(parent) + 1:
            inds = []
            
            for i in last:
                inds += list(np.where(parent == i)[0] + 1)
                
            sample_order += last
            last = inds
        
        return sample_order  
    
    def generate_new_sample(self, sample_size):
        """Generate new sample from probability density
        
        Args:
        sample_size: int. Size of sample to be generated.
           
        Returns:
        new_sample: array. Numpy array containing new sample.
        """
         # Initialize new sample matrix
        new_sample = np.zeros([sample_size, self.length])
        
        # Create random matrix
        rand_sample = np.random.uniform(size = [sample_size, self.length])
        
        # Get value of first bit in new samples
        new_sample[np.where(rand_sample[:, 0] < self.probs[0, 0])[0], 0] = 1
        
        # Get sample order
        sample_order = self.find_sample_order(self.parent)[1:]
    
        # Get values for remaining bits in new samples
        for i in sample_order:
            par_ind = self.parent[i-1]
            probs = np.zeros(sample_size)
            
            ind_0 = np.where(new_sample[:, par_ind] == 0)[0]
            ind_1 = np.where(new_sample[:, par_ind] == 1)[0]
            
            probs[ind_0] = self.probs[0, i]
            probs[ind_1] = self.probs[1, i]
            
            new_sample[np.where(rand_sample[:, i] < probs)[0], i] = 1
        
        return new_sample
        
        
