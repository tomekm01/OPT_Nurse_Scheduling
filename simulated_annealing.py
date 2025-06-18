# simulated_annealing.py

import random
import math
from problem import create_random_schedule, evaluate

# SA parameters
INITIAL_TEMP = 100000
MAX_ITERATIONS = 2000

#Randomly modifies a shift for a single nurse on a single day.
def tweak_schedule(schedule):
    new_schedule = [row[:] for row in schedule]
    nurse = random.randint(0, len(new_schedule) - 1)
    day = random.randint(0, len(new_schedule[0]) - 1)
    new_shift = random.choice([0, 1, 2])
    new_schedule[nurse][day] = new_shift
    return new_schedule

def cooling_linear(iteration):
    alpha = 1
    return INITIAL_TEMP / (1 + alpha * iteration)


def run_simulated_annealing():
    current_plan = create_random_schedule()
    best_plan = [row[:] for row in current_plan]
    current_temp = INITIAL_TEMP

    for k in range(MAX_ITERATIONS):
        candidate = tweak_schedule(current_plan)

        curr_score = evaluate(current_plan)
        new_score = evaluate(candidate)

        if new_score < curr_score or random.random() < math.exp((curr_score - new_score) / current_temp):
            current_plan = candidate

        if evaluate(current_plan) < evaluate(best_plan):
            best_plan = [row[:] for row in current_plan]

        current_temp = cooling_linear(k)

    return best_plan

