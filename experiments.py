import random
import statistics
import time
from collections import Counter

import matplotlib.pyplot as plt

import problem
from ant_colony import run_ant_colony, run_ant_colony_improved
from genetic import run_genetic_algorithm
from simulated_annealing import run_simulated_annealing
from tabu_search import run_tabu_search


NUM_RUNS = 10
BASE_SEED = 12345


INDIVIDUAL_ALGORITHMS = [
    {
        "label": "SA-Exp-Tuned",
        "func": run_simulated_annealing,
        "args": {
            "exponential": True,
            "exp_alpha": 0.85,
            "initial_temp": 100_000,
            "max_iterations": 2000,
        },
    },
    {
        "label": "Tabu-Tuned",
        "func": run_tabu_search,
        "args": {
            "tabu_tenure": 2,
            "neighborhood_size": 20,
        },
    },
    {
        "label": "Genetic-Tuned",
        "func": run_genetic_algorithm,
        "args": {
            "population_size": 100,
            "mutation_rate": 0.02,
        },
    },
    {
        "label": "ACO-Tuned",
        "func": run_ant_colony,
        "args": {
            "num_ants": 100,
            "num_iterations": 200,
            "evaporation": 0.9,
            "alpha": 1.0,
            "beta": 2.0,
        },
    },
    {
        "label": "ACO-Improved",
        "func": run_ant_colony_improved,
        "args": {
            "num_ants": 100,
            "num_iterations": 200,
            "evaporation": 0.9,
            "alpha": 1.0,
            "beta": 2.0,
        },
    },
]


PORTFOLIOS = [
    {
        "label": "Fast-Portfolio",
        "candidates": ["SA-Exp-Tuned", "Tabu-Tuned", "Genetic-Tuned"],
    },
    {
        "label": "Full-Portfolio-Baseline",
        "candidates": [
            "SA-Exp-Tuned",
            "Tabu-Tuned",
            "Genetic-Tuned",
            "ACO-Tuned",
        ],
    },
    {
        "label": "Full-Portfolio-Improved",
        "candidates": [
            "SA-Exp-Tuned",
            "Tabu-Tuned",
            "Genetic-Tuned",
            "ACO-Improved",
        ],
    },
]


PROBLEM_SIZES = [
    ("Medium", 30, 20),
    ("Large", 50, 30),
]


def run_algorithm(algorithm, seed):
    args = dict(algorithm["args"])
    args["seed"] = seed

    start = time.time()
    schedule = algorithm["func"](**args)
    score = problem.evaluate(schedule)
    runtime_s = round(time.time() - start, 4)

    return {
        "schedule": schedule,
        "score": score,
        "runtime_s": runtime_s,
    }


def build_portfolio(portfolio, run_results):
    candidate_results = [
        (name, run_results[name])
        for name in portfolio["candidates"]
    ]

    # The portfolio selects the lowest penalty score. If scores tie, it chooses
    # the fastest tied algorithm so the rule is deterministic and reportable.
    winner_name, winner_result = min(
        candidate_results,
        key=lambda item: (item[1]["score"], item[1]["runtime_s"]),
    )
    runtime_s = round(
        sum(result["runtime_s"] for _, result in candidate_results),
        4,
    )

    return {
        "schedule": winner_result["schedule"],
        "score": winner_result["score"],
        "runtime_s": runtime_s,
        "winner": winner_name,
    }


def add_result(results, label, score, runtime_s):
    results[label]["scores"].append(score)
    results[label]["runtimes"].append(runtime_s)


def print_summary(size_label, results, all_labels):
    print(f"\n===== {size_label} Final Summary =====")
    print(
        f"{'Method':25} | {'Mean Score':>10} | {'Std Score':>9} | "
        f"{'Min':>8} | {'Max':>8} | {'Mean Time':>10} | {'Std Time':>9}"
    )
    print("-" * 99)

    for label in all_labels:
        scores = results[label]["scores"]
        runtimes = results[label]["runtimes"]
        print(
            f"{label:25} | "
            f"{statistics.mean(scores):10.2f} | "
            f"{statistics.pstdev(scores):9.2f} | "
            f"{min(scores):8} | "
            f"{max(scores):8} | "
            f"{statistics.mean(runtimes):10.4f} | "
            f"{statistics.pstdev(runtimes):9.4f}"
        )


def print_winner_frequencies(winner_counts):
    print("\nPortfolio Winner Frequencies")
    print(f"{'Portfolio':25} | {'Internal Algorithm':18} | {'Wins':>4}")
    print("-" * 53)

    for portfolio in PORTFOLIOS:
        label = portfolio["label"]
        for candidate in portfolio["candidates"]:
            print(
                f"{label:25} | "
                f"{candidate:18} | "
                f"{winner_counts[label].get(candidate, 0):4}"
            )


def save_plots(size_label, results, all_labels, winner_counts):
    x = range(len(all_labels))
    mean_scores = [statistics.mean(results[label]["scores"]) for label in all_labels]
    mean_times = [statistics.mean(results[label]["runtimes"]) for label in all_labels]

    plt.figure(figsize=(11, 5))
    plt.bar(x, mean_scores)
    plt.xticks(x, all_labels, rotation=30, ha="right")
    plt.ylabel("Mean Penalty Score")
    plt.title(f"{size_label}: Research Project 2 Mean Score")
    plt.tight_layout()
    plt.savefig(f"{size_label}_rp2_mean_score.png")
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.bar(x, mean_times)
    plt.xticks(x, all_labels, rotation=30, ha="right")
    plt.ylabel("Mean Runtime (s)")
    plt.title(f"{size_label}: Research Project 2 Mean Runtime")
    plt.tight_layout()
    plt.savefig(f"{size_label}_rp2_mean_time.png")
    plt.close()

    internal_labels = [algorithm["label"] for algorithm in INDIVIDUAL_ALGORITHMS]
    x = range(len(internal_labels))
    width = 0.25
    offsets = [-width, 0, width]

    plt.figure(figsize=(11, 5))
    for offset, portfolio in zip(offsets, PORTFOLIOS):
        label = portfolio["label"]
        counts = [
            winner_counts[label].get(internal_label, 0)
            for internal_label in internal_labels
        ]
        plt.bar(
            [i + offset for i in x],
            counts,
            width,
            label=label,
        )

    plt.xticks(x, internal_labels, rotation=30, ha="right")
    plt.ylabel("Winner Count")
    plt.title(f"{size_label}: Portfolio Winner Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{size_label}_rp2_winner_frequency.png")
    plt.close()


def run_experiments():
    individual_labels = [
        algorithm["label"]
        for algorithm in INDIVIDUAL_ALGORITHMS
    ]
    portfolio_labels = [
        portfolio["label"]
        for portfolio in PORTFOLIOS
    ]
    all_labels = individual_labels + portfolio_labels

    for size_label, nurses, days in PROBLEM_SIZES:
        problem.NUM_NURSES = nurses
        problem.NUM_DAYS = days

        print(f"\n===== {size_label} ({nurses} nurses x {days} days) =====")

        results = {
            label: {"scores": [], "runtimes": []}
            for label in all_labels
        }
        winner_counts = {
            portfolio["label"]: Counter()
            for portfolio in PORTFOLIOS
        }

        for run_id in range(1, NUM_RUNS + 1):
            seed = BASE_SEED + run_id
            random.seed(seed)
            print(f"\n--- Run {run_id}/{NUM_RUNS} (seed={seed}) ---")

            run_results = {}

            for algorithm in INDIVIDUAL_ALGORITHMS:
                label = algorithm["label"]
                result = run_algorithm(algorithm, seed)
                run_results[label] = result
                add_result(results, label, result["score"], result["runtime_s"])

                print(
                    f"{label:25} | Score: {result['score']:8} | "
                    f"Time: {result['runtime_s']:8.4f}s"
                )

            for portfolio in PORTFOLIOS:
                label = portfolio["label"]
                result = build_portfolio(portfolio, run_results)
                add_result(results, label, result["score"], result["runtime_s"])
                winner_counts[label][result["winner"]] += 1

                print(
                    f"{label:25} | Score: {result['score']:8} | "
                    f"Time: {result['runtime_s']:8.4f}s | "
                    f"Winner: {result['winner']}"
                )

        print_summary(size_label, results, all_labels)
        print_winner_frequencies(winner_counts)
        save_plots(size_label, results, all_labels, winner_counts)

    print("\nResearch Project 2 experiment complete.")
    print("Plots saved to the project directory.")


if __name__ == "__main__":
    run_experiments()
