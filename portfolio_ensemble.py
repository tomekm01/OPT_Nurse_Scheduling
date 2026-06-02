import time

import problem
from ant_colony import run_ant_colony
from genetic import run_genetic_algorithm
from simulated_annealing import run_simulated_annealing
from tabu_search import run_tabu_search


FULL_PORTFOLIO_ALGORITHMS = [
    (
        "SA-Exp-Tuned",
        run_simulated_annealing,
        {
            "exponential": True,
            "exp_alpha": 0.85,
            "initial_temp": 100_000,
            "max_iterations": 2000,
        },
    ),
    (
        "Tabu-Tuned",
        run_tabu_search,
        {
            "tabu_tenure": 2,
            "neighborhood_size": 20,
        },
    ),
    (
        "Genetic-Tuned",
        run_genetic_algorithm,
        {
            "population_size": 100,
            "mutation_rate": 0.02,
        },
    ),
    (
        "ACO-Tuned",
        run_ant_colony,
        {
            "num_ants": 100,
            "num_iterations": 200,
            "evaporation": 0.9,
            "alpha": 1.0,
            "beta": 2.0,
        },
    ),
]

FAST_PORTFOLIO_ALGORITHMS = FULL_PORTFOLIO_ALGORITHMS[:3]


def _run_portfolio(algorithms, seed=None):
    candidates = {}
    best_schedule = None
    best_score = float("inf")
    winning_algorithm = None

    # The portfolio runs several metaheuristics and selects the schedule with
    # the lowest penalty according to the existing project evaluation function.
    for name, func, args in algorithms:
        run_args = dict(args)
        if seed is not None:
            run_args["seed"] = seed

        start = time.time()
        schedule = func(**run_args)
        runtime_s = round(time.time() - start, 4)
        score = problem.evaluate(schedule)

        candidates[name] = {
            "score": score,
            "runtime_s": runtime_s,
        }

        if score < best_score:
            best_schedule = schedule
            best_score = score
            winning_algorithm = name

    return best_schedule, best_score, winning_algorithm, candidates


def run_full_portfolio(seed=None):
    """
    Run SA, Tabu Search, Genetic Algorithm, and ACO with tuned parameters.

    Returns:
        best_schedule, best_score, winning_algorithm, candidates
    """
    return _run_portfolio(FULL_PORTFOLIO_ALGORITHMS, seed=seed)


def run_fast_portfolio(seed=None):
    """
    Run the faster tuned portfolio: SA, Tabu Search, and Genetic Algorithm.

    ACO is intentionally excluded because it was much slower in prior
    experiments.

    Returns:
        best_schedule, best_score, winning_algorithm, candidates
    """
    return _run_portfolio(FAST_PORTFOLIO_ALGORITHMS, seed=seed)
