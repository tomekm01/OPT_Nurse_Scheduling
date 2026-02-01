import time
import statistics
import matplotlib.pyplot as plt
import numpy as np
import problem
from tabu_search import run_tabu_search

NUM_RUNS = 10

# Defaults for the parameter NOT being tested
DEF_NEIGHBORHOOD = 20 
DEF_TENURE       = 10

experiments = [
    {
        'label': 'Tenure-Small (2)',
        'args':  { 'max_iterations': 1000, 'tabu_tenure': 2,  'neighborhood_size': DEF_NEIGHBORHOOD }
    },
    {
        'label': 'Tenure-Med (10)',
        'args':  { 'max_iterations': 1000, 'tabu_tenure': 10, 'neighborhood_size': DEF_NEIGHBORHOOD }
    },
    {
        'label': 'Tenure-Big (50)',
        'args':  { 'max_iterations': 1000, 'tabu_tenure': 50, 'neighborhood_size': DEF_NEIGHBORHOOD }
    },

    {
        'label': 'Neigh-Small (5)',
        'args':  { 'max_iterations': 1000, 'tabu_tenure': DEF_TENURE, 'neighborhood_size': 5 }
    },
    {
        'label': 'Neigh-Med (20)',
        'args':  { 'max_iterations': 1000, 'tabu_tenure': DEF_TENURE, 'neighborhood_size': DEF_NEIGHBORHOOD }
    },
    {
        'label': 'Neigh-Big (50)',
        'args':  { 'max_iterations': 1000, 'tabu_tenure': DEF_TENURE, 'neighborhood_size': 50 }
    },
]

problem_sizes = [
    ("Small",  5,  7),
    ("Medium", 30, 20),
    ("Large",  50, 30),
]

for size_label, nurses, days in problem_sizes:
    problem.NUM_NURSES = nurses
    problem.NUM_DAYS   = days

    print(f"\n===== {size_label} ({nurses}×{days}) =====")

    results = {exp['label']: {'scores': [], 'times': []} for exp in experiments}

    for run in range(1, NUM_RUNS + 1):
        print(f"\n--- Run {run} ---")

        for exp in experiments:
            label = exp['label']
            args = exp['args']

            t0 = time.time()
            
            schedule = run_tabu_search(seed=run, **args)
            
            elapsed = round(time.time() - t0, 2)
            score = problem.evaluate(schedule)

            results[label]['scores'].append(score)
            results[label]['times'].append(elapsed)

            print(f"{label:20} → Score: {score:6} | Time: {elapsed:6}s")

    print("\n--- Summary ---")
    for label, data in results.items():
        sc = data['scores']
        tm = data['times']
        print(
            f"{label:20} | "
            f"Score → mean={statistics.mean(sc):.2f}, "
            f"std={statistics.pstdev(sc):.2f}, "
            f"min={min(sc)}, max={max(sc)} | "
            f"Time → mean={statistics.mean(tm):.2f}s"
        )

    labels = list(results.keys())
    mean_scores = [statistics.mean(results[l]['scores']) for l in labels]
    mean_times = [statistics.mean(results[l]['times']) for l in labels]
    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4']*3 + ['#ff7f0e']*3 
    bars = plt.bar(x, mean_scores, color=colors)
    
    for bar, value in zip(bars, mean_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{value:.1f}', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Mean Penalty Score")
    plt.title(f"{size_label} – Tabu Search Tuning (Blue=Tenure, Orange=Neighbor)")
    plt.tight_layout()
    plt.savefig(f"{size_label}_mean_score_TABU_tuning.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    bars = plt.bar(x, mean_times, color=colors)
    
    for bar, value in zip(bars, mean_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{value:.2f}s', 
                 ha='center', va='bottom', fontsize=9)

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Mean Runtime (s)")
    plt.title(f"{size_label} – Tabu Search Runtime")
    plt.tight_layout()
    plt.savefig(f"{size_label}_mean_time_TABU_tuning.png")
    plt.close()