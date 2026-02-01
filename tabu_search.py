import random
from problem import create_random_schedule, evaluate


def run_tabu_search(
    *,
    max_iterations=1000,
    tabu_tenure=10,
    neighborhood_size=20,
    seed=None
):
    if seed is not None:
        random.seed(seed)

    # Initial solution
    current = create_random_schedule()
    curr_score = evaluate(current)

    best = [row[:] for row in current]
    best_score = curr_score

    # tabu_list: move -> expiration iteration
    tabu_list = {}

    for k in range(max_iterations):

        local_best_candidate = None
        local_best_score = float('inf')
        local_best_move = None

        seen_moves = set()

        # Sample a neighborhood WITHOUT duplicate moves
        while len(seen_moves) < neighborhood_size:
            candidate, move = tweak_schedule_with_move_info(current)

            if move in seen_moves:
                continue

            seen_moves.add(move)
            score = evaluate(candidate)

            is_tabu = move in tabu_list and tabu_list[move] > k
            is_aspiration = score < best_score

            if not is_tabu or is_aspiration:
                if score < local_best_score:
                    local_best_candidate = candidate
                    local_best_score = score
                    local_best_move = move

        # Move to the selected neighbor
        if local_best_candidate is not None:
            current = local_best_candidate
            curr_score = local_best_score

            # Add reverse move to tabu list
            tabu_list[local_best_move] = k + tabu_tenure

            if curr_score < best_score:
                best_score = curr_score
                best = [row[:] for row in current]

        # Periodic cleanup of expired tabu moves
        if k % 100 == 0:
            tabu_list = {m: exp for m, exp in tabu_list.items() if exp > k}

    return best


def tweak_schedule_with_move_info(schedule):
    """
    Single-cell reassignment with full move encoding:
    (nurse, day, old_shift, new_shift)
    """
    new_schedule = [row[:] for row in schedule]

    nurse_idx = random.randint(0, len(new_schedule) - 1)
    day_idx = random.randint(0, len(new_schedule[0]) - 1)

    old_shift = new_schedule[nurse_idx][day_idx]
    possible_shifts = [s for s in (0, 1, 2) if s != old_shift]
    new_shift = random.choice(possible_shifts)

    new_schedule[nurse_idx][day_idx] = new_shift

    move = (nurse_idx, day_idx, old_shift, new_shift)
    return new_schedule, move
