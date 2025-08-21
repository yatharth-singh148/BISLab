import random

# 1. Define the Problem: The function to maximize
# We'll use a simple function: f(x) = -(x^2) + 10x + 100
# The genetic algorithm will find the integer value of x that maximizes this function
def fitness_function(x):
    return -(x**2) + 10 * x + 100

# 2. Initialize Parameters
population_size = 50
mutation_rate = 0.05
crossover_rate = 0.7
num_generations = 100
chromosome_length = 10  # This determines the range of x values (2^10 = 1024, so x can be 0-1023)
def binary_to_int(binary_string):
    return int(binary_string, 2)
def int_to_binary(number, length):
    return format(number, f'0{length}b')
def create_initial_population(size, length):
    population = []
    for _ in range(size):
        individual = ''.join(random.choice(['0', '1']) for _ in range(length))
        population.append(individual)
    return population
def evaluate_fitness(population, func):
    fitness_scores = []
    for individual in population:
        x = binary_to_int(individual)
        score = func(x)
        fitness_scores.append(score)
    return fitness_scores
def selection(population, fitness_scores):
    selected = []
    for _ in range(len(population)):
        tournament_size = 3
        tournament_candidates = random.sample(list(zip(population, fitness_scores)), tournament_size)
        winner = max(tournament_candidates, key=lambda item: item[1])
        selected.append(winner[0])
    return selected
def crossover(parent1, parent2, rate):
    if random.random() < rate:
        point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:point] + parent2[point:]
        offspring2 = parent2[:point] + parent1[point:]
        return offspring1, offspring2
    else:
        return parent1, parent2
def mutate(individual, rate):
    mutated_individual = list(individual)
    for i in range(len(mutated_individual)):
        if random.random() < rate:
            mutated_individual[i] = '1' if mutated_individual[i] == '0' else '0'
    return ''.join(mutated_individual)
def run_genetic_algorithm():
    population = create_initial_population(population_size, chromosome_length)
    best_solution = None
    best_fitness = float('-inf')

    for generation in range(num_generations):
        fitness_scores = evaluate_fitness(population, fitness_function)
        current_best_index = fitness_scores.index(max(fitness_scores))
        current_best_individual = population[current_best_index]
        current_best_fitness = fitness_scores[current_best_index]

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_individual
        selected_population = selection(population, fitness_scores)
        next_generation = []
        for i in range(0, population_size, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i+1]
            offspring1, offspring2 = crossover(parent1, parent2, crossover_rate)
            next_generation.append(mutate(offspring1, mutation_rate))
            next_generation.append(mutate(offspring2, mutation_rate))
        
        population = next_generation
        print(f"Generation {generation+1}: Best fitness = {best_fitness:.2f} (x = {binary_to_int(best_solution)})")
    print("\n--- Optimization Complete ---")
    x_value = binary_to_int(best_solution)
    print(f"The best solution found is x = {x_value}")
    print(f"The maximum value of the function is: {fitness_function(x_value)}")
if __name__ == '__main__':
    run_genetic_algorithm()
