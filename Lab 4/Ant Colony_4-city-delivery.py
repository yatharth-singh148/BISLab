import numpy as np

# --- 1. Define the Problem ---
# City coordinates (A, B, C, D)
coords = np.array([
    [0, 0],  # A: Warehouse
    [1, 3],  # B: Customer 1
    [4, 1],  # C: Customer 2
    [5, 4]   # D: Customer 3
])
num_cities = len(coords)

# Calculate distance matrix (Euclidean distance)
def distance(city1, city2):
    return np.sqrt(np.sum((city1 - city2)**2))

dist_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(i + 1, num_cities):
        d = distance(coords[i], coords[j])
        dist_matrix[i, j] = dist_matrix[j, i] = d

# --- 2. Initialize Parameters ---
num_ants = 5
num_iterations = 10
alpha = 1.0  # Pheromone importance
beta = 2.0   # Heuristic importance (distance)
rho = 0.5    # Evaporation rate

# Initial pheromone matrix (equal for all edges)
initial_pheromone = 1.0 / (num_cities * np.mean(dist_matrix[dist_matrix > 0]))
pheromone_matrix = np.ones((num_cities, num_cities)) * initial_pheromone

# Pre-calculate heuristic information (visibility: 1/distance)
heuristic_matrix = 1.0 / (dist_matrix + 1e-10) # Avoid division by zero
np.fill_diagonal(heuristic_matrix, 0)

# Track the best solution found
best_route = None
best_length = np.inf

print(f"Starting ACO with {num_ants} ants for {num_cities} cities over {num_iterations} iterations...")

# --- 5. Iterate ---
for iteration in range(num_iterations):
    
    # List to store the routes and lengths found by each ant
    ant_solutions = []

    # --- 3. Construct Solutions ---
    for ant in range(num_ants):
        # Start at a random city (or fixed start like City 0/Warehouse A)
        start_city = 0
        current_route = [start_city]
        unvisited = list(range(num_cities))
        unvisited.remove(start_city)
        
        while unvisited:
            current_city = current_route[-1]
            
            # Calculate probabilities for the next city
            pheromones = pheromone_matrix[current_city, unvisited]**alpha
            heuristics = heuristic_matrix[current_city, unvisited]**beta
            
            # Attractiveness = (Pheromone^alpha) * (Heuristic^beta)
            attractiveness = pheromones * heuristics
            
            # Selection Probability
            probabilities = attractiveness / np.sum(attractiveness)
            
            # Select the next city probabilistically
            next_city_index = np.random.choice(len(unvisited), 1, p=probabilities)[0]
            next_city = unvisited[next_city_index]
            
            current_route.append(next_city)
            unvisited.remove(next_city)
        
        # Complete the tour by returning to the start city
        current_route.append(start_city) 
        
        # Calculate the tour length
        tour_length = 0
        for i in range(num_cities):
            city1 = current_route[i]
            city2 = current_route[i+1]
            tour_length += dist_matrix[city1, city2]
            
        ant_solutions.append((current_route, tour_length))

        # --- 6. Output the Best Solution (Track) ---
        if tour_length < best_length:
            best_length = tour_length
            best_route = current_route
    
    # --- 4. Update Pheromones ---
    
    # Evaporation (Pheromone reduction)
    pheromone_matrix *= (1 - rho)
    
    # Deposition (Pheromone increase)
    for route, length in ant_solutions:
        # Pheromone deposit (Q/L), where Q=1 is often used, L is length
        deposit = 1.0 / length
        for i in range(num_cities):
            city1 = route[i]
            city2 = route[i+1]
            pheromone_matrix[city1, city2] += deposit
            pheromone_matrix[city2, city1] += deposit # TSP is symmetric

print("-" * 30)
print(f"Final Best Route: {best_route}")
print(f"Final Best Length: {best_length:.2f} units")
# Map city indices back to names for clarity
city_names = {0: 'A (Warehouse)', 1: 'B (Cust 1)', 2: 'C (Cust 2)', 3: 'D (Cust 3)'}
named_route = [city_names[c] for c in best_route]
print(f"Named Route: {named_route}")
