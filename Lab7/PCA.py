import numpy as np

# --- 1. Define the Problem (distance-based fitness) ---
stores = np.array([[2, 3], [10, 15], [8, 5]])  # store locations

def fitness(cell_pos):
    # total distance to all stores
    return np.sum(np.linalg.norm(stores - cell_pos, axis=1))

# --- 2. Initialize Grid ---
grid_size = 10
iterations = 20

# randomly place warehouse candidates on a grid
cells = np.random.randint(0, 20, size=(grid_size, grid_size, 2))

# --- Helper to get neighbors ---
def neighbors(i, j):
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    for di, dj in dirs:
        if 0 <= i+di < grid_size and 0 <= j+dj < grid_size:
            yield (i+di, j+dj)

# --- 3. Iterate: Update States ---
for _ in range(iterations):
    new_cells = np.copy(cells)
    for i in range(grid_size):
        for j in range(grid_size):
            # choose the best among the cell and its neighbors
            candidates = [(cells[i,j], fitness(cells[i,j]))]
            for ni, nj in neighbors(i, j):
                candidates.append((cells[ni, nj], fitness(cells[ni, nj])))

            # pick the cell with lowest fitness
            best_pos, _ = min(candidates, key=lambda x: x[1])
            new_cells[i,j] = best_pos

    cells = new_cells

# --- 4. Output the Best Solution ---
all_positions = cells.reshape(-1, 2)
best_location = min(all_positions, key=fitness)

print("Best warehouse location found:", best_location)
print("Total distance to stores:", fitness(best_location))
