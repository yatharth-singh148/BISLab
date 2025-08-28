import random
teachers = ["Alice", "Bob", "Charlie", "Diana"]
subjects = ["Math", "Science", "History", "English"]
preferences = {
    "Alice": {"Math": 9, "Science": 7, "History": 4, "English": 5},
    "Bob": {"Math": 6, "Science": 8, "History": 7, "English": 6},
    "Charlie": {"Math": 5, "Science": 6, "History": 9, "English": 7},
    "Diana": {"Math": 7, "Science": 5, "History": 6, "English": 8}
}
POP_SIZE = 10
GENERATIONS = 30
MUTATION_RATE = 0.2
def random_solution():
    return random.sample(subjects, len(teachers))

population = [random_solution() for _ in range(POP_SIZE)]
def fitness(solution):
    score = 0
    for teacher, subject in zip(teachers, solution):
        score += preferences[teacher][subject]
    return score
def selection(population):
    sorted_pop = sorted(population, key=fitness, reverse=True)
    return sorted_pop[:2]
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    child = parent1[:point] + [s for s in parent2 if s not in parent1[:point]]
    return child
def mutate(solution):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(solution)), 2)
        solution[i], solution[j] = solution[j], solution[i]
    return solution
best_solution = None
for gen in range(GENERATIONS):
    new_population = []
    for _ in range(POP_SIZE):
        parent1, parent2 = selection(population)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population.append(child)
    population = new_population
    best_solution = max(population, key=fitness)
    print(f"Generation {gen+1}: Best Score = {fitness(best_solution)}")
print("\nBest Assignment Found:")
for teacher, subject in zip(teachers, best_solution):
    print(f"{teacher} → {subject} (Score: {preferences[teacher][subject]})")

print("Total Fitness:", fitness(best_solution))
