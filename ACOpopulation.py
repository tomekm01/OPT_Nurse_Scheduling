import time, statistics
import problem
import matplotlib.pyplot as plt

from simulated_annealing import run_simulated_annealing
from ant_colony       import run_ant_colony

NUM_RUNS = 10

# different parameters
experiments = [
    {
        'label': 'ACO',
        'func':  run_ant_colony,
        'args':  {
            'num_ants':       40,
            'num_iterations': 500,
            'evaporation':    0.9,
            'alpha':          1.0,
            'beta':           2.0
        }
    },
    {
        'label': 'ACO20300',
        'func':  run_ant_colony,
        'args':  {
            'num_ants':       20,
            'num_iterations': 300,
            'evaporation':    0.9,
            'alpha':          1.0,
            'beta':           2.0
        }
    },
    {
        'label': 'ACO30200',
        'func':  run_ant_colony,
        'args':  {
            'num_ants':       30,
            'num_iterations': 200,
            'evaporation':    0.9,
            'alpha':          1.0,
            'beta':           2.0
        }
    },
]

# test sizes
problem_sizes = [
    ("Small",  5,  7),
    ("Medium", 30, 20),
    ("Large", 50,30),
]

for size_label, nurses, days in problem_sizes:
    problem.NUM_NURSES = nurses
    problem.NUM_DAYS   = days

    print(f"\n===== {size_label} ({nurses}×{days}) =====")

    # prepare storage
    results = {exp['label']: {'scores':[], 'times':[]} for exp in experiments}

    for run in range(1, NUM_RUNS+1):
        print(f"\n--- Run {run} ---")
        for exp in experiments:
            label = exp['label']
            func  = exp['func']
            args  = exp['args']

            t0 = time.time()
            schedule = func(**args)
            score    = problem.evaluate(schedule)
            elapsed  = round(time.time() - t0, 2)

            results[label]['scores'].append(score)
            results[label]['times'].append(elapsed)

            print(f"{label:10} → Score: {score:6} | Time: {elapsed:6}s")

    # summary
    print("\n--- Summary ---")
    for label, data in results.items():
        sc, tm = data['scores'], data['times']
        print(f"{label:10} | "
              f"Score → mean={statistics.mean(sc):.2f}, std={statistics.pstdev(sc):.2f}, "
              f"min={min(sc)}, max={max(sc)} | "
              f"Time  → mean={statistics.mean(tm):.2f}s, std={statistics.pstdev(tm):.2f}s, "
              f"min={min(tm)}s, max={max(tm)}s")

    # --------------- plotting ----------------
    labels = list(results.keys())
    mean_scores = [statistics.mean(results[lbl]['scores']) for lbl in labels]
    mean_times  = [statistics.mean(results[lbl]['times'])  for lbl in labels]
    x = list(range(len(labels)))
    import numpy as np
    # Mean Score plot
    plt.figure()
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    bars = plt.bar(x, mean_scores, color=colors)
    for bar, value in zip(bars, mean_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,
                 height + 0.01 * max(mean_scores),
                 f'{value:.2f}',
                 ha='center',
                 va='bottom',
                 fontsize=9)
    plt.xticks(x, labels, rotation=45)
    plt.ylabel('Mean Penalty Score')
    plt.title(f'{size_label} Mean Score')
    plt.tight_layout()
    plt.savefig(f'{size_label}_mean_score_ACOpopulation.png')
    plt.close()

    # Mean Time plot
    plt.figure()
    bars = plt.bar(x, mean_times, color=colors)
    for bar, value in zip(bars, mean_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,
                 height + 0.01 * max(mean_times),
                 f'{value:.4f}',
                 ha='center',
                 va='bottom',
                 fontsize=9)
    plt.xticks(x, labels, rotation=45)
    plt.ylabel('Mean Runtime (s)')
    plt.title(f'{size_label} Mean Time')
    plt.tight_layout()
    plt.savefig(f'{size_label}_mean_time_ACOpopulation.png')
    plt.close()
