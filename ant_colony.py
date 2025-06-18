# ant_colony.py

import random
import problem

# ACO parameters (tweakable)
NUM_ANTS       = 40
NUM_ITERATIONS = 500
EVAPORATION    = 0.9
ALPHA          = 1
BETA           = 2

def run_ant_colony():
    n_nurses = problem.NUM_NURSES
    n_days   = problem.NUM_DAYS
    shifts   = problem.SHIFTS

    # initialize pheromones
    pheromones = {
        (n, d, s): 1.0
        for n in range(n_nurses)
        for d in range(n_days)
        for s in shifts
    }

    best_schedule = None
    best_score    = float('inf')

    for _ in range(NUM_ITERATIONS):
        # each ant builds a solution
        for _ in range(NUM_ANTS):
            schedule = [[None]*n_days for _ in range(n_nurses)]
            for d in range(n_days):
                for n in range(n_nurses):
                    choices = []
                    for s in shifts:
                        tau = pheromones[(n, d, s)] ** ALPHA
                        eta = 1.0  # uniform heuristic (could refine later)
                        choices.append((s, tau * (eta**BETA)))

                    total = sum(weight for _, weight in choices)
                    r     = random.random() * total
                    cum   = 0
                    for s, weight in choices:
                        cum += weight
                        if r <= cum:
                            schedule[n][d] = s
                            break

            score = problem.evaluate(schedule)
            if score < best_score:
                best_score, best_schedule = score, [row[:] for row in schedule]

        # pheromone evaporation
        for key in pheromones:
            pheromones[key] *= EVAPORATION

        # reinforce best
        for n in range(n_nurses):
            for d, s in enumerate(best_schedule[n]):
                pheromones[(n, d, s)] += 1.0 / (1 + best_score)

    return best_schedule
