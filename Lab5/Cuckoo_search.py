import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# --- 1. Define the Problem (Objective Function) ---
def objective_function(r):
    """Minimize the surface area of a cylinder with V=1000. r must be positive."""
    if r <= 0:
        return np.inf  # Penalize non-positive radius
    return 2 * np.pi * r**2 + 2000 / r

# --- 2. Initialize Parameters ---
n = 10          # Number of host nests (proposed radii)
[cite_start]Pa = 0.2        # Probability of discovery/abandonment [cite: 36]
max_t = 50      # Maximum number of iterations
search_range = [1.0, 20.0] # Practical bounds for radius r (in cm)
[cite_start]beta = 1.5      # Levy flight parameter [cite: 48]
[cite_start]alpha = 0.01    # Step size scaling factor [cite: 50]

# --- Helper function for Levy Flight (Simplified) ---
def levy_flight(x_current, alpha, beta):
    """Generates a new step using a simplified Levy Flight calculation."""
    # Source: Mantegna's algorithm constants for step size
    sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
               (np.math.gamma((1 + beta) / 2) * beta * (2**((beta - 1) / 2))))**(1 / beta)
    
    u = np.random.normal(0, sigma_u)
    v = np.random.normal(0, 1)
    step = alpha * (u / (np.abs(v)**(1 / beta)))
    
    x_new = x_current + step
    
    # Boundary check: ensure the new radius stays within bounds
    x_new = np.clip(x_new, search_range[0], search_range[1])
    
    return x_new

# --- 3. Initialize Population ---
# Generate initial host nests (solutions: radii r) randomly
nests = np.random.uniform(search_range[0], search_range[1], size=n)
fitness = [objective_function(r) for r in nests]

# Track the best solution
best_index = np.argmin(fitness)
best_solution_r = nests[best_index]
best_fitness_A = fitness[best_index]

print(f"Starting CS for Cylinder Cost Minimization (Volume = 1000 cm³).")
print(f"Initial Best Radius (r): {best_solution_r:.4f} cm | Area (f(r)): {best_fitness_A:.4f} cm²")

# --- 7. Iterate ---
for t in range(max_t):
    
    # --- 5. Generate New Solution (Cuckoo) via Lévy Flight ---
    k = np.random.randint(n) # Pick one nest to base the cuckoo egg on
    r_new = levy_flight(nests[k], alpha, beta)
    f_new = objective_function(r_new)
    
    # Compare with a randomly chosen host nest 'j'
    j = np.random.randint(n) 
    
    # Select Best: If the cuckoo's new fitness is superior (lower area), replace the host's nest
    if f_new < fitness[j]:
        nests[j] = r_new
        fitness[j] = f_new
        
        # Update overall best
        if f_new < best_fitness_A:
            best_solution_r = r_new
            best_fitness_A = f_new

    # --- 6. Abandon Worst Nests ---
    # [cite_start]Abandon a fraction Pa of the worst nests and replace them [cite: 91, 92]
    num_abandon = int(np.ceil(Pa * n))
    sorted_indices = np.argsort(fitness)
    
    for i in range(num_abandon):
        # We only abandon if a random roll is less than Pa
        if np.random.rand() < Pa: 
            worst_index = sorted_indices[n - 1 - i]
            
            # Replace with a new random solution (often a local random walk, or fully random for high exploration)
            # Here, we use a fully random solution in the search space:
            new_r_random = np.random.uniform(search_range[0], search_range[1])
            new_f_random = objective_function(new_r_random)
            
            nests[worst_index] = new_r_random
            fitness[worst_index] = new_f_random
            
            # Update best if this new random solution is better
            if new_f_random < best_fitness_A:
                best_solution_r = new_r_random
                best_fitness_A = new_f_random

    if (t + 1) % 10 == 0:
        print(f"Iter {t+1}: Current Best r={best_solution_r:.4f} cm | Area={best_fitness_A:.4f} cm²")

# --- 8. Output the Best Solution ---
# Calculate the corresponding height for the best radius
optimal_h = 1000 / (np.pi * best_solution_r**2)
theoretical_optimal_r = np.cbrt(500/np.pi) # The mathematically proven optimal r is approx 5.419

print("-" * 50)
print("CS Final Results:")
print(f"Optimal Radius (r): {best_solution_r:.4f} cm")
print(f"Optimal Height (h): {optimal_h:.4f} cm")
print(f"Minimum Surface Area: {best_fitness_A:.4f} cm²")

# Note: The true mathematical optimum occurs when r = h.
print(f"Optimal r (theoretical): {theoretical_optimal_r:.4f} cm")
