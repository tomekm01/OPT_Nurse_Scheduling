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


def run_ant_colony_improved(
    *,
    num_ants       = 100,
    num_iterations = 200,
    evaporation    = 0.9,
    alpha          = 1.0,
    beta           = 2.0,
    seed           = None
):
    """
    Improved ACO for the nurse scheduling problem.

    Returns a best_schedule (list[list[int]]), matching run_ant_colony().
    """
    if seed is not None:
        random.seed(seed)

    n_nurses = problem.NUM_NURSES
    n_days   = problem.NUM_DAYS
    shifts   = problem.SHIFTS
    evaluate = problem.evaluate

    pheromones = {
        (n, d, s): 1.0
        for n in range(n_nurses)
        for d in range(n_days)
        for s in shifts
    }

    best_schedule = None
    best_score    = float('inf')
    min_eta       = 0.01

    for _ in range(num_iterations):
        iteration_best_schedule = None
        iteration_best_score    = float('inf')

        for _ in range(num_ants):
            schedule = [[None] * n_days for _ in range(n_nurses)]

            for d in range(n_days):
                shift_counts = {s: 0 for s in shifts}

                for n in range(n_nurses):
                    weights = []
                    for s in shifts:
                        tau = pheromones[(n, d, s)] ** alpha

                        # Heuristic value eta uses scheduling knowledge:
                        # 1) avoid Night-to-Morning violations,
                        # 2) prefer shifts that still need coverage today.
                        eta = 1.0

                        if (
                            d > 0
                            and schedule[n][d - 1] == 2
                            and s == 0
                        ):
                            eta *= min_eta

                        missing_coverage = (
                            problem.MIN_COVERAGE_PER_SHIFT - shift_counts[s]
                        )
                        if missing_coverage > 0:
                            eta *= 1.0 + missing_coverage
                        else:
                            eta *= 0.5

                        eta = max(min_eta, eta)
                        weights.append((s, tau * (eta ** beta)))

                    total = sum(w for _, w in weights)
                    if total <= 0:
                        selected_shift = random.choice(shifts)
                    else:
                        r = random.random() * total
                        cum = 0.0
                        selected_shift = shifts[-1]
                        for s, w in weights:
                            cum += w
                            if r <= cum:
                                selected_shift = s
                                break

                    schedule[n][d] = selected_shift
                    shift_counts[selected_shift] += 1

            # Simple repair: remove Night-to-Morning violations by replacing
            # the Morning shift with the better of Evening or Night.
            repair_night_to_morning(schedule)

            score = evaluate(schedule)
            if score < iteration_best_score:
                iteration_best_score    = score
                iteration_best_schedule = [row[:] for row in schedule]

            if score < best_score:
                best_score    = score
                best_schedule = [row[:] for row in schedule]

        for key in pheromones:
            pheromones[key] *= evaporation

        # Reinforce good schedules. Lower penalty scores receive a larger
        # pheromone deposit through 1 / (1 + score).
        if iteration_best_schedule is not None:
            deposit = 1.0 / (1.0 + iteration_best_score)
            for n in range(n_nurses):
                for d, s in enumerate(iteration_best_schedule[n]):
                    pheromones[(n, d, s)] += deposit

        if best_schedule is not None:
            deposit = 1.0 / (1.0 + best_score)
            for n in range(n_nurses):
                for d, s in enumerate(best_schedule[n]):
                    pheromones[(n, d, s)] += deposit

    return best_schedule


def repair_night_to_morning(schedule):
    for nurse in range(len(schedule)):
        for day in range(1, len(schedule[0])):
            if schedule[nurse][day - 1] == 2 and schedule[nurse][day] == 0:
                best_shift = None
                best_score = float('inf')

                for replacement in [s for s in problem.SHIFTS if s != 0]:
                    schedule[nurse][day] = replacement
                    score = problem.evaluate(schedule)
                    if score < best_score:
                        best_score = score
                        best_shift = replacement

                schedule[nurse][day] = best_shift
