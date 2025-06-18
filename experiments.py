import time
from simulated_annealing import run_simulated_annealing
from problem import evaluate

NUM_RUNS = 10

for run in range(1, NUM_RUNS + 1):
    print(f"\n--- Run {run} ---")
    start_time = time.time()

    best_schedule = run_simulated_annealing()
    score = evaluate(best_schedule)

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    print(f"Score: {score}")
    print(f"Time: {duration} seconds")
