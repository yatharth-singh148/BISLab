import random

# ----------------------------
# Particle Swarm Optimization
# ----------------------------

class Particle:
    def __init__(self, dim, bounds):
        self.position = [random.randint(bounds[i][0], bounds[i][1]) for i in range(dim)]
        self.velocity = [0] * dim
        self.best_position = list(self.position)
        self.best_score = float('inf')

    def update_velocity(self, global_best, w=0.5, c1=1.5, c2=1.5):
        for i in range(len(self.velocity)):
            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best[i] - self.position[i])
            self.velocity[i] = int(w * self.velocity[i] + cognitive + social)

    def update_position(self, bounds):
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]
            # Keep inside bounds
            self.position[i] = max(bounds[i][0], min(bounds[i][1], self.position[i]))

# ----------------------------
# Real-life Example: Staff Scheduling
# ----------------------------

# Suppose 3 workers (Alice, Bob, Charlie)
# Each wants: Alice=3 shifts, Bob=4, Charlie=2 (per week)
preferences = [3, 4, 2]

# Total shifts needed per week = 9 (3 per day for 3 days example)
total_shifts = sum(preferences)

# Bounds: each worker can have between 0 and total_shifts
bounds = [(0, total_shifts)] * len(preferences)

# Fitness function: how well does a schedule match preferences?
def fitness(schedule):
    # Must assign exactly total_shifts
    if sum(schedule) != total_shifts:
        return float('inf')  # invalid
    # Minimize squared error from preferences
    return sum((schedule[i] - preferences[i])**2 for i in range(len(schedule)))

# ----------------------------
# Run PSO
# ----------------------------
num_particles = 30
num_iterations = 50

# Initialize swarm
swarm = [Particle(len(preferences), bounds) for _ in range(num_particles)]
global_best = None
global_best_score = float('inf')

for iteration in range(num_iterations):
    for particle in swarm:
        score = fitness(particle.position)

        # Update personal best
        if score < particle.best_score:
            particle.best_score = score
            particle.best_position = list(particle.position)

        # Update global best
        if score < global_best_score:
            global_best_score = score
            global_best = list(particle.position)

    # Update velocity & position
    for particle in swarm:
        particle.update_velocity(global_best)
        particle.update_position(bounds)

print("Best schedule found:", global_best)
print("Score (lower is better):", global_best_score)
print("Preferences:", preferences)