# tuned_experiments.py

import time, statistics
import random
import problem
import matplotlib.pyplot as plt

from simulated_annealing import run_simulated_annealing
from ant_colony       import run_ant_colony
from utils_results    import append_row, json_params

NUM_RUNS = 10
BASE_SEED = 12345

# Tuned algorithms with their “best” parameters
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

# Final problem sizes
problem_sizes = [
    ("Small",  5,  7),
    ("Medium", 30, 20),
    ("Large",  50, 30),
]

RAW_CSV = "results/tuned_raw.csv"

for size_label, nurses, days in problem_sizes:
    problem.NUM_NURSES = nurses
    problem.NUM_DAYS   = days

    print(f"\n===== {size_label} ({nurses}×{days}) =====")

    results = {exp['label']:{'scores':[], 'times':[]} for exp in experiments}

    for run in range(1, NUM_RUNS+1):
        seed = BASE_SEED + run

        print(f"\n--- Run {run} (seed={seed}) ---")

        # also seed the global RNG used by problem.create_random_schedule etc.
        random.seed(seed)

        for exp in experiments:
            lbl, fn, args = exp['label'], exp['func'], exp['args']

            # pass seed down to algorithms (works because we added seed param)
            run_args = dict(args)
            run_args["seed"] = seed

            t0 = time.time()
            schedule = fn(**run_args)
            score    = problem.evaluate(schedule)
            dt       = round(time.time() - t0, 4)

            results[lbl]['scores'].append(score)
            results[lbl]['times'].append(dt)

            append_row(RAW_CSV, {
                "size": size_label,
                "nurses": nurses,
                "days": days,
                "run_id": run,
                "seed": seed,
                "algo": lbl,
                "params": json_params(args),
                "score": score,
                "runtime_s": dt
            })

            print(f"{lbl:10} → Score: {score:8} | Time: {dt:8.4f}s")

    # summary
    print("\n--- Summary ---")
    for lbl, data in results.items():
        sc, tm = data['scores'], data['times']
        print(f"{lbl:10} | "
              f"Score → μ={statistics.mean(sc):.2f}, σ={statistics.pstdev(sc):.2f}, "
              f"median={statistics.median(sc):.2f}, [{min(sc)}, {max(sc)}] | "
              f"Time  → μ={statistics.mean(tm):.4f}s, σ={statistics.pstdev(tm):.4f}s, "
              f"[{min(tm):.4f}s, {max(tm):.4f}s]")

    # plots (same style as you had)
    labels      = list(results.keys())
    mean_scores = [statistics.mean(results[l]['scores']) for l in labels]
    mean_times  = [statistics.mean(results[l]['times'])  for l in labels]
    x           = range(len(labels))

    plt.figure()
    plt.bar(x, mean_scores)
    plt.xticks(x, labels, rotation=45)
    plt.ylabel('Mean Penalty Score')
    plt.title(f'{size_label}: Mean Score')
    plt.tight_layout()
    plt.savefig(f'{size_label}_tuned_mean_score.png')
    plt.close()

    plt.figure()
    plt.bar(x, mean_times)
    plt.xticks(x, labels, rotation=45)
    plt.ylabel('Mean Runtime (s)')
    plt.title(f'{size_label}: Mean Time')
    plt.tight_layout()
    plt.savefig(f'{size_label}_tuned_mean_time.png')
    plt.close()

print(f"\nRaw results saved to: {RAW_CSV}")
print("Next: run statistical significance test:")
print("  python stats_significance.py results/tuned_raw.csv")
