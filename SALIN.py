import time, statistics
import problem
import matplotlib.pyplot as plt

from simulated_annealing import run_simulated_annealing
from ant_colony       import run_ant_colony

NUM_RUNS = 10

# different parameters
experiments = [
    {
        'label': 'SA-Linear.100.1',
        'func':  run_simulated_annealing,
        'args':  {
            'exponential':False,
            'initial_temp':100_000,
            'max_iterations':2000,
            'linear_alpha':1.0
        }
    },
    {
        'label': 'SA-Linear.50.1',
        'func': run_simulated_annealing,
        'args': {
            'exponential': False,
            'initial_temp': 50_000,
            'max_iterations': 2000,
            'linear_alpha': 1.0
        }
    },
    {
        'label': 'SA-Linear.200.1',
        'func': run_simulated_annealing,
        'args': {
            'exponential': False,
            'initial_temp': 200_000,
            'max_iterations': 2000,
            'linear_alpha': 1.0
        }
    },
    {
        'label': 'SA-Linear.100.08',
        'func': run_simulated_annealing,
        'args': {
            'exponential': False,
            'initial_temp': 100_000,
            'max_iterations': 2000,
            'linear_alpha': 0.8
        }
    },
    {
        'label': 'SA-Linear.50.08',
        'func': run_simulated_annealing,
        'args': {
            'exponential': False,
            'initial_temp': 50_000,
            'max_iterations': 2000,
            'linear_alpha': 0.8
        }
    },
    {
        'label': 'SA-Linear.200.08',
        'func': run_simulated_annealing,
        'args': {
            'exponential': False,
            'initial_temp': 200_000,
            'max_iterations': 2000,
            'linear_alpha': 0.8
        }
    },
    {
        'label': 'SA-Linear.100.06',
        'func': run_simulated_annealing,
        'args': {
            'exponential': False,
            'initial_temp': 100_000,
            'max_iterations': 2000,
            'linear_alpha': 0.6
        }
    },
    {
        'label': 'SA-Linear.50.06',
        'func': run_simulated_annealing,
        'args': {
            'exponential': False,
            'initial_temp': 50_000,
            'max_iterations': 2000,
            'linear_alpha': 0.6
        }
    },
    {
        'label': 'SA-Linear.200.06',
        'func': run_simulated_annealing,
        'args': {
            'exponential': False,
            'initial_temp': 200_000,
            'max_iterations': 2000,
            'linear_alpha': 0.6
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

    # Mean Score plot
    plt.figure()
    plt.bar(x, mean_scores)
    plt.xticks(x, labels, rotation=45)
    plt.ylabel('Mean Penalty Score')
    plt.title(f'{size_label} Mean Score')
    plt.tight_layout()
    plt.savefig(f'{size_label}_mean_score_SALIN.png')
    plt.close()

    # Mean Time plot
    plt.figure()
    plt.bar(x, mean_times)
    plt.xticks(x, labels, rotation=45)
    plt.ylabel('Mean Runtime (s)')
    plt.title(f'{size_label} Mean Time')
    plt.tight_layout()
    plt.savefig(f'{size_label}_mean_time_SALIN.png')
    plt.close()

    # Line Plot: Score vs Run Number
    plt.figure()
    for label in labels:
        plt.plot(range(1, NUM_RUNS + 1), results[label]['scores'], marker='o', label=label)

