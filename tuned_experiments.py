# tuned_experiments.py

import time, statistics
import problem
import matplotlib.pyplot as plt

from simulated_annealing import run_simulated_annealing
from ant_colony       import run_ant_colony

NUM_RUNS = 10

# 1) Tuned algorithms with their “best” parameters
experiments = [
    {
        'label': 'SA-Linear',
        'func':  run_simulated_annealing,
        'args':  {
            'exponential': False,
            'initial_temp': 50_000,
            'max_iterations': 2000,
            'linear_alpha': 1.0
        }
    },
    {
        'label': 'SA-Exp',
        'func':  run_simulated_annealing,
        'args':  {
            'exponential': True,
            'exp_alpha': 0.85,
            'initial_temp': 100_000,
            'max_iterations': 2000
        }
    },
    {
        'label': 'ACO-Tuned',
        'func':  run_ant_colony,
        'args':  {
            'num_ants':       100,
            'num_iterations': 200,
            'evaporation':    0.9,
            'alpha':          1.0,
            'beta':           2.0
        }
    },
]

# 2) Final problem sizes from the report
problem_sizes = [
    ("Small",  5,  7),
    ("Medium", 30, 20),
    ("Large",  50, 30),
]

for size_label, nurses, days in problem_sizes:
    problem.NUM_NURSES = nurses
    problem.NUM_DAYS   = days

    print(f"\n===== {size_label} ({nurses}×{days}) =====")

    # storage
    results = {exp['label']:{'scores':[], 'times':[]} for exp in experiments}

    # run each algorithm NUM_RUNS times
    for run in range(1, NUM_RUNS+1):
        print(f"\n--- Run {run} ---")
        for exp in experiments:
            lbl, fn, args = exp['label'], exp['func'], exp['args']
            t0 = time.time()
            schedule = fn(**args)
            score    = problem.evaluate(schedule)
            dt       = round(time.time() - t0, 2)

            results[lbl]['scores'].append(score)
            results[lbl]['times'].append(dt)
            print(f"{lbl:10} → Score: {score:6} | Time: {dt:6}s")

    # print summary
    print("\n--- Summary ---")
    for lbl, data in results.items():
        sc, tm = data['scores'], data['times']
        print(f"{lbl:10} | "
              f"Score → μ={statistics.mean(sc):.2f}, σ={statistics.pstdev(sc):.2f}, "
              f"[{min(sc)}, {max(sc)}] | "
              f"Time  → μ={statistics.mean(tm):.2f}s, σ={statistics.pstdev(tm):.2f}s, "
              f"[{min(tm)}s, {max(tm)}s]")

    # plot & save
    labels      = list(results.keys())
    mean_scores = [statistics.mean(results[l]['scores']) for l in labels]
    mean_times  = [statistics.mean(results[l]['times'])  for l in labels]
    x           = range(len(labels))

    # Score plot
    plt.figure()
    plt.bar(x, mean_scores)
    plt.xticks(x, labels, rotation=45)
    plt.ylabel('Mean Penalty Score')
    plt.title(f'{size_label}: Mean Score')
    plt.tight_layout()
    plt.savefig(f'{size_label}_tuned_mean_score.png')
    plt.close()

    # Time plot
    plt.figure()
    plt.bar(x, mean_times)
    plt.xticks(x, labels, rotation=45)
    plt.ylabel('Mean Runtime (s)')
    plt.title(f'{size_label}: Mean Time')
    plt.tight_layout()
    plt.savefig(f'{size_label}_tuned_mean_time.png')
    plt.close()
