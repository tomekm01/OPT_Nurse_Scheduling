import sys
import os
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np

import problem
from genetic import run_genetic_algorithm

NUM_RUNS = 5

experiments = [
    {
        'label': 'GA-Pop.50-Mut.02',
        'func':  run_genetic_algorithm,
        'args':  { 'generations': 100, 'population_size': 50, 'mutation_rate': 0.02 }
    },
    {
        'label': 'GA-Pop.50-Mut.05',
        'func':  run_genetic_algorithm,
        'args':  { 'generations': 100, 'population_size': 50, 'mutation_rate': 0.05 }
    },
    {
        'label': 'GA-Pop.100-Mut.02',
        'func':  run_genetic_algorithm,
        'args':  { 'generations': 100, 'population_size': 100, 'mutation_rate': 0.02 }
    },
    {
        'label': 'GA-Pop.20-Mut.05',
        'func':  run_genetic_algorithm,
        'args':  { 'generations': 200, 'population_size': 20, 'mutation_rate': 0.05 }
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

            print(f"{label:18} → Score: {score:6} | Time: {elapsed:6}s")

    print("\n--- Summary ---")
    for label, data in results.items():
        sc, tm = data['scores'], data['times']
        print(f"{label:18} | "
              f"Score → mean={statistics.mean(sc):.2f}, min={min(sc)} | "
              f"Time  → mean={statistics.mean(tm):.2f}s")

    labels = list(results.keys())
    mean_scores = [statistics.mean(results[lbl]['scores']) for lbl in labels]
    mean_times  = [statistics.mean(results[lbl]['times'])  for lbl in labels]
    x = np.arange(len(labels))

    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(labels)))
    bars = plt.bar(x, mean_scores, color=colors)
    
    for bar, value in zip(bars, mean_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,
                 height + 0.01 * max(mean_scores),
                 f'{value:.1f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel('Mean Penalty Score')
    plt.title(f'{size_label} - Genetic Algorithm Performance')
    plt.tight_layout()
    plt.savefig(f'{size_label}_mean_score_GA.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, mean_times, color=colors)
    
    for bar, value in zip(bars, mean_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,
                 height + 0.01 * max(mean_times),
                 f'{value:.2f}s',
                 ha='center', va='bottom', fontsize=9)

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel('Mean Runtime (s)')
    plt.title(f'{size_label} - Genetic Algorithm Runtime')
    plt.tight_layout()
    plt.savefig(f'{size_label}_mean_time_GA.png')
    plt.close()