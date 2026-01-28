# ant_colony.py

import random
import problem

def run_ant_colony(
    *,
    num_ants       = 40,
    num_iterations = 500,
    evaporation    = 0.9,
    alpha          = 1.0,
    beta           = 2.0,
    seed           = None
):
    """
    Returns a best_schedule (list[list[int]]).
    Added: seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

    # grab the up-to-date problem dimensions
    n_nurses = problem.NUM_NURSES
    n_days   = problem.NUM_DAYS
    shifts   = problem.SHIFTS
    evaluate = problem.evaluate

    # initialize pheromones
    pheromones = {
        (n, d, s): 1.0
        for n in range(n_nurses)
        for d in range(n_days)
        for s in shifts
    }

    best_schedule = None
    best_score    = float('inf')

    for _ in range(num_iterations):
        # each ant builds a schedule
        for _ in range(num_ants):
            schedule = [[None] * n_days for _ in range(n_nurses)]
            for d in range(n_days):
                for n in range(n_nurses):
                    weights = []
                    for s in shifts:
                        tau = pheromones[(n, d, s)] ** alpha
                        eta = 1.0
                        weights.append((s, tau * (eta ** beta)))

                    total = sum(w for _, w in weights)
                    if total <= 0:
                        # fallback uniform choice if weights degenerate
                        schedule[n][d] = random.choice(shifts)
                        continue

                    r   = random.random() * total
                    cum = 0.0
                    for s, w in weights:
                        cum += w
                        if r <= cum:
                            schedule[n][d] = s
                            break
                    if schedule[n][d] is None:
                        schedule[n][d] = shifts[-1]

            score = evaluate(schedule)
            if score < best_score:
                best_score    = score
                best_schedule = [row[:] for row in schedule]

        # evaporate pheromones
        for key in pheromones:
            pheromones[key] *= evaporation

        # reinforce the global best (safe-guard)
        if best_schedule is not None:
            for n in range(n_nurses):
                for d, s in enumerate(best_schedule[n]):
                    pheromones[(n, d, s)] += 1.0 / (1.0 + best_score)

    return best_schedule
