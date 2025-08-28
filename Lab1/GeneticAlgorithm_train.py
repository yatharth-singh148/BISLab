import random

# -----------------------------
# Train Station Optimization
# -----------------------------

# Example: maximize passenger satisfaction depending on trains allocated
# Suppose assigning x trains gives satisfaction = -(x^2) + 10x + 100
def fitness(trains: int) -> int:
    return -(trains**2) + 10 * trains + 100

# Parameters
POP_SIZE = 50
GENS = 100
MUT_RATE = 0.05
CROSS_RATE = 0.7
CHROM_LEN = 10  # binary string length (x in 0–1023)

# Helpers
to_int = lambda b: int(b, 2)
to_bin = lambda n: format(n, f'0{CHROM_LEN}b')

def init_pop(size): return [''.join(random.choice("01") for _ in range(CHROM_LEN)) for _ in range(size)]
def evaluate(pop): return [fitness(to_int(ind)) for ind in pop]

def select(pop, fits):
    return [max(random.sample(list(zip(pop, fits)), 3), key=lambda t: t[1])[0] for _ in pop]

def crossover(p1, p2):
    if random.random() < CROSS_RATE:
        pt = random.randint(1, CHROM_LEN - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
    return p1, p2

def mutate(ind):
    return ''.join('1' if (g=='0' and random.random()<MUT_RATE) else 
                   '0' if (g=='1' and random.random()<MUT_RATE) else g for g in ind)

def run_ga():
    pop, best_fit, best_sol = init_pop(POP_SIZE), float('-inf'), None

    for gen in range(1, GENS+1):
        fits = evaluate(pop)
        best_idx = max(range(len(fits)), key=fits.__getitem__)
        if fits[best_idx] > best_fit:
            best_fit, best_sol = fits[best_idx], pop[best_idx]

        selected = select(pop, fits)
        next_gen = []
        for i in range(0, POP_SIZE, 2):
            c1, c2 = crossover(selected[i], selected[i+1])
            next_gen += [mutate(c1), mutate(c2)]
        pop = next_gen

        print(f"Gen {gen:03}: Best satisfaction = {best_fit} (Trains = {to_int(best_sol)})")

    print("\n--- Optimization Complete ---")
    x = to_int(best_sol)
    print(f"Optimal trains allocation = {x}")
    print(f"Max passenger satisfaction = {fitness(x)}")

if __name__ == "__main__":
    run_ga()