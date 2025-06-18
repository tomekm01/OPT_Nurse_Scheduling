# problem.py

import random

# Constants
NUM_NURSES = 10
NUM_DAYS = 20
SHIFTS = [0, 1, 2]  # 0: Morning, 1: Evening, 2: Night
OFF_SHIFT = -1
MAX_CONSECUTIVE_WORK_DAYS = 7
MIN_COVERAGE_PER_SHIFT = 2  # At least 2 nurses per shift per day

def create_random_schedule():
    """Generates a random initial schedule for all nurses."""
    return [[random.choice(SHIFTS) for _ in range(NUM_DAYS)] for _ in range(NUM_NURSES)]

def evaluate(schedule):
    """Evaluates a schedule based on hard and soft constraint violations."""
    hard_penalty = 0
    soft_penalty = 0

    # Hard constraint: No night shift followed by morning shift
    for nurse in range(NUM_NURSES):
        for day in range(NUM_DAYS - 1):
            if schedule[nurse][day] == 2 and schedule[nurse][day + 1] == 0:
                hard_penalty += 1

    # Soft constraint: Max 7 consecutive working days
    for nurse in range(NUM_NURSES):
        consecutive_work = 0
        for day in range(NUM_DAYS):
            if schedule[nurse][day] != OFF_SHIFT:
                consecutive_work += 1
                if consecutive_work > MAX_CONSECUTIVE_WORK_DAYS:
                    soft_penalty += 1
            else:
                consecutive_work = 0

    # Hard constraint: Minimum coverage per shift per day
    for day in range(NUM_DAYS):
        shift_counts = {s: 0 for s in SHIFTS}
        for nurse in range(NUM_NURSES):
            shift = schedule[nurse][day]
            if shift in SHIFTS:
                shift_counts[shift] += 1
        for s in SHIFTS:
            if shift_counts[s] < MIN_COVERAGE_PER_SHIFT:
                hard_penalty += (MIN_COVERAGE_PER_SHIFT - shift_counts[s])

    return hard_penalty * 100 + soft_penalty * 10
