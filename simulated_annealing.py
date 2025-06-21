import random, math
from problem import create_random_schedule, evaluate

def cooling_linear(T0, alpha, k):
    return T0 / (1 + alpha * k)

def cooling_exponential(T0, alpha, k):
    return T0 * (alpha ** k)

def run_simulated_annealing(
    *,
    initial_temp   = 100_000,
    max_iterations = 2_000,
    linear_alpha   = 1.0,
    exp_alpha      = 0.85,
    exponential    = False
):
    """
    Runs SA with either linear or exponential cooling.
    """
    current = create_random_schedule()
    best    = [row[:] for row in current]

    for k in range(max_iterations):
        # pick temperature schedule
        T = (cooling_exponential(initial_temp, exp_alpha, k)
             if exponential
             else cooling_linear(initial_temp, linear_alpha, k))

        # tweak
        candidate   = tweak_schedule(current)
        curr_score  = evaluate(current)
        new_score   = evaluate(candidate)

        # acceptance
        if new_score < curr_score or random.random() < math.exp((curr_score - new_score) / T):
            current = candidate

        # update best
        if evaluate(current) < evaluate(best):
            best = [row[:] for row in current]

    return best


def tweak_schedule(schedule):
    new_schedule = [row[:] for row in schedule]
    nurse = random.randint(0, len(new_schedule) - 1)
    day   = random.randint(0, len(new_schedule[0]) - 1)
    new_shift = random.choice([0, 1, 2])
    new_schedule[nurse][day] = new_shift
    return new_schedule
