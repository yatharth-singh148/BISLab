import numpy as np
import random
COST_COEFFS = {
    'plant1': {'a': 0.004, 'b': 2.0, 'c': 300},
    'plant2': {'a': 0.006, 'b': 1.5, 'c': 500},
    'plant3': {'a': 0.009, 'b': 1.8, 'c': 400}
}
PLANTS = list(COST_COEFFS.keys())
POWER_DEMAND = 150.0

MIN_BOUNDS = np.array([10.0, 20.0, 15.0])
MAX_BOUNDS = np.array([80.0, 100.0, 90.0])

# Penalty for not meeting the power demand
PENALTY_FACTOR = 1000.0

def economic_load_dispatch_fitness(position):
    total_cost = 0
    for i, plant_name in enumerate(PLANTS):
        P = position[i]  # Power for this plant
        coeffs = COST_COEFFS[plant_name]
        cost = coeffs['a'] * (P**2) + coeffs['b'] * P + coeffs['c']
        total_cost += cost
    total_power_generated = np.sum(position)
    demand_error = abs(total_power_generated - POWER_DEMAND)
    fitness = total_cost + (PENALTY_FACTOR * demand_error)
    
    return fitness
num_wolves = 20       # Number of wolves (solutions) in the pack
dim = len(PLANTS)     # Number of dimensions (3, one for each plant)
max_iterations = 100  # How many times we update the pack
positions = np.zeros((num_wolves, dim))
for i in range(num_wolves):
    
    positions[i, :] = np.random.uniform(MIN_BOUNDS, MAX_BOUNDS, dim)

alpha_pos = np.zeros(dim)
alpha_score = float("inf")  

beta_pos = np.zeros(dim)
beta_score = float("inf")

delta_pos = np.zeros(dim)
delta_score = float("inf")

print(f"Running GWO for Economic Load Dispatch...")
print(f"Total Power Demand: {POWER_DEMAND} MW")
print(f"Optimizing for {dim} power plants over {max_iterations} iterations.")

# --- 6. Iterate ---
for it in range(max_iterations):
    
    # --- 4. Evaluate Fitness ---
    for i in range(num_wolves):
        # Ensure wolf positions are within bounds before evaluating
        positions[i, :] = np.clip(positions[i, :], MIN_BOUNDS, MAX_BOUNDS)
        
        # Calculate the fitness (cost) for the current wolf
        fitness = economic_load_dispatch_fitness(positions[i, :])
        
        # Update Alpha, Beta, and Delta
        if fitness < alpha_score:
            delta_score, delta_pos = beta_score, beta_pos.copy()
            beta_score, beta_pos = alpha_score, alpha_pos.copy()
            alpha_score, alpha_pos = fitness, positions[i, :].copy()
        elif fitness < beta_score:
            delta_score, delta_pos = beta_score, beta_pos.copy()
            beta_score, beta_pos = fitness, positions[i, :].copy()
        elif fitness < delta_score:
            delta_score, delta_pos = fitness, positions[i, :].copy()

    a = 2 - it * (2 / max_iterations)
    for i in range(num_wolves):
        # Update based on Alpha
        r1, r2 = random.random(), random.random()
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = abs(C1 * alpha_pos - positions[i, :])
        X1 = alpha_pos - A1 * D_alpha

        # Update based on Beta
        r1, r2 = random.random(), random.random()
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = abs(C2 * beta_pos - positions[i, :])
        X2 = beta_pos - A2 * D_beta

        # Update based on Delta
        r1, r2 = random.random(), random.random()
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = abs(C3 * delta_pos - positions[i, :])
        X3 = delta_pos - A3 * D_delta

        # New position is the average of the three updates
        positions[i, :] = (X1 + X2 + X3) / 3
        positions[i, :] = np.clip(positions[i, :], MIN_BOUNDS, MAX_BOUNDS)

    if (it + 1) % 20 == 0:
        print(f"Iteration {it + 1}: Best Cost (Alpha Score) = ${alpha_score:.2f}")

print("\nGWO optimization finished.")

# Recalculate final cost and power for clarity
final_cost = 0
for i, plant_name in enumerate(PLANTS):
    P = alpha_pos[i]
    coeffs = COST_COEFFS[plant_name]
    cost = coeffs['a'] * (P**2) + coeffs['b'] * P + coeffs['c']
    final_cost += cost
    print(f"  - Plant {i+1} Output: {P:.2f} MW")

total_gen = np.sum(alpha_pos)
print(f"\nTotal Power Generated: {total_gen:.2f} MW (Demand: {POWER_DEMAND} MW)")
print(f"Total Generation Cost: ${final_cost:.2f}")
print(f"Best Fitness (Cost + Penalty): ${alpha_score:.2f}")
