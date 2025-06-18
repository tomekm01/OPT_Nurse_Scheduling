import time
import statistics

import problem
from simulated_annealing import run_simulated_annealing
from ant_colony       import run_ant_colony

NUM_RUNS = 10

# Define the test‐case sizes you want to run:
problem_sizes = [
    ("Small",  5,  7),
    ("Medium", 50, 30),
    # (“Large”, 200,300),  # uncomment if you’re happy waiting a while
]

for label, n, d in problem_sizes:
    # override problem dimensions
    problem.NUM_NURSES = n
    problem.NUM_DAYS   = d

    print(f"\n===== {label} Problem ({n} nurses × {d} days) =====")

    sa_scores,  sa_times  = [], []
    aco_scores, aco_times = [], []

    for run in range(1, NUM_RUNS + 1):
        print(f"\n--- Run {run} ---")

        # Simulated Annealing
        t0   = time.time()
        best = run_simulated_annealing()
        s    = problem.evaluate(best)
        ta   = round(time.time() - t0, 2)
        print(f"SA  → Score: {s} | Time: {ta}s")
        sa_scores.append(s)
        sa_times.append(ta)

        # Ant Colony Optimization
        t0   = time.time()
        best = run_ant_colony()
        s    = problem.evaluate(best)
        tb   = round(time.time() - t0, 2)
        print(f"ACO → Score: {s} | Time: {tb}s")
        aco_scores.append(s)
        aco_times.append(tb)

    # summary function
    def summarize(name, scores, times):
        print(f"\n{name} Summary:")
        print(f"  • Score →  mean={statistics.mean(scores):.2f}, "
              f"std={statistics.pstdev(scores):.2f}, "
              f"min={min(scores)}, max={max(scores)}")
        print(f"  • Time  →  mean={statistics.mean(times):.2f}s, "
              f"std={statistics.pstdev(times):.2f}s, "
              f"min={min(times)}s, max={max(times)}s")

    summarize("Simulated Annealing", sa_scores, sa_times)
    summarize("Ant Colony Optimization", aco_scores, aco_times)
