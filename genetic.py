import numpy as np

from plotter import ImagePlotter

from logger import Logger


class GeneticAlgorithm:
    def __init__(self, pop_size=100, max_gen=100, mut_rate=0.01, fitness_function='euclidean_distance'):
        self.population_size = pop_size
        self.max_generations = max_gen
        self.mutation_rate = mut_rate

        self.latest_prediction, self.latest_fitness = None, 0
        self.best_prediction, self.best_fitness = None, 0

        self.population = None

        self.fitness_scores = None
        self.fitness_function = fitness_function

        self.image = None


    def fit(self, image, x, y):
        self.image = image

        # Initialize population
        self.population = self._init_population()

        # logger = Logger('logs.csv')

        for generation in range(self.max_generations):
            # Fitness calculation
            self.fitness_scores = [self._calculate_fitness(p) for p in self.population]

            # Roulette wheel selection
            selection = self._roulette_wheel_selection()

            # Crossover
            self.population = self._crossover_population(selection)

            # Mutate and update population
            self.population = [self._mutate(p) for p in self.population]

            # Calculate latest prediction and fitness
            self.latest_fitness = self._calculate_fitness(self.latest_prediction) if self.latest_prediction is not None else 0
            if self.latest_prediction is not None:
                print(f'Generation {generation + 1}\nCurrent prediction: {self.latest_prediction}\nCurrent fitness: {self.latest_fitness}\n')
                logger.log(generation, self.population_size, self.max_generations, self.mutation_rate, self.fitness_function, self.latest_fitness)

            # Update best prediction
            self.latest_prediction = self.population[np.argmax(self.fitness_scores)]
            if self.best_prediction is None or self.latest_fitness > self.best_fitness:
                self.best_prediction = self.latest_prediction
                self.best_fitness = self.latest_fitness

            # Display predictions
            plt = ImagePlotter(900, 300, plot_name=f'Generation {generation + 1}')
            plt.draw_img(self.image, x, y)
            plt.draw_img(self.best_prediction, x + 300, y)
            plt.draw_img(self.latest_prediction, x + 600, y)
            plt.display(delay=10)


    def _cosine_similarity(self, p):
        cosine_similarities = []
        for p_value, image_value in zip(p, self.image):
            p_vectors = np.array([np.cos(np.radians(p_value)), np.sin(np.radians(p_value))])
            image_vectors = np.array([np.cos(np.radians(image_value)), np.sin(np.radians(image_value))])
            cosine_similarity = np.dot(p_vectors, image_vectors) / (np.linalg.norm(p_vectors) * np.linalg.norm(image_vectors))
            cosine_similarities.append(cosine_similarity)
        return np.mean(cosine_similarities)
    

    def _euclidean_distance(self, p):
        euclidean_distances = []
        for p_value, image_value in zip(p, self.image):
            p_vectors = np.array([np.cos(np.radians(p_value)), np.sin(np.radians(p_value))])
            image_vectors = np.array([np.cos(np.radians(image_value)), np.sin(np.radians(image_value))])
            euclidean_distance = np.linalg.norm(p_vectors - image_vectors)
            euclidean_distances.append(euclidean_distance)
        return np.mean(euclidean_distances)
    

    def _mean_absolute_error(self, p):
        mae_err = []
        for p_value, image_value in zip(p, self.image):
            mae_err.append(abs(p_value - image_value))
        return np.mean(mae_err)
    

    def _calculate_fitness(self, p):
        if self.fitness_function == 'cosine_similarity':
            cosine_similarity = self._cosine_similarity(p)
            return cosine_similarity if cosine_similarity > 0 else 0 # ensure positive values
        
        if self.fitness_function == 'euclidean_distance':
            euclidean_distance = self._euclidean_distance(p)
            euclidean_similarity = np.inf if euclidean_distance == 0 else 1 / euclidean_distance # to achieve maximization
            return euclidean_similarity

        if self.fitness_function == 'mean_absolute_error':
            mae_err = self._mean_absolute_error(p)
            mae_similarity = np.inf if mae_similarity == 0 else 1 / mae_err # to achieve maximization
            return mae_similarity
        
        raise ValueError('Invalid similarity function specified.')


    def _roulette_wheel_selection(self):
        total_fitness = sum(self.fitness_scores)
        prob = [score / total_fitness for score in self.fitness_scores] # higher fitness -> higher probability
        selection = np.random.choice(range(len(self.fitness_scores)), size=self.population_size, p=prob)
        return selection


    def _crossover(self, parent1, parent2):
        index = np.random.randint(0, len(parent1))
        child1 = np.concatenate((parent1[:index], parent2[index:]))
        child2 = np.concatenate((parent2[:index], parent1[index:]))
        return child1, child2


    def _crossover_population(self, selection):
        new_population = []
        for i in range(0, self.population_size, 2):
                parent1 = self.population[selection[i]]
                parent2 = self.population[selection[i + 1]]
                child1, child2 = self._crossover(parent1, parent2)
                new_population.append(child1)
                new_population.append(child2)
        return new_population
    

    def _mutate(self, p):
        mutant = p.copy()
        for i in range(len(p)):
            if np.random.random() < self.mutation_rate:
                mutant[i] = np.random.randint(0, 360)
        return mutant
    

    def _init_population(self):
        population = []
        for _ in range(self.population_size):
            p = np.random.randint(0, 360, size=len(self.image))
            population.append(p)
        return population