import random
from problem import create_random_schedule, evaluate

def run_genetic_algorithm(
    *,
    generations     = 100,
    population_size = 50,
    crossover_rate  = 0.8,
    mutation_rate   = 0.02,
    tournament_size = 3,
    elitism         = True,
    seed            = None
):

    if seed is not None:
        random.seed(seed)

    population = [create_random_schedule() for _ in range(population_size)]
    
    ranked_population = []
    for indiv in population:
        ranked_population.append((evaluate(indiv), indiv))
    
    ranked_population.sort(key=lambda x: x[0])
    
    best_overall_score = ranked_population[0][0]
    best_overall_schedule = [row[:] for row in ranked_population[0][1]]

    for gen in range(generations):
        next_generation = []
        
        if elitism:
            best_current = [row[:] for row in ranked_population[0][1]]
            next_generation.append(best_current)

        while len(next_generation) < population_size:
            parent1 = tournament_selection(ranked_population, tournament_size)
            parent2 = tournament_selection(ranked_population, tournament_size)
            
            if random.random() < crossover_rate:
                offspring = crossover_uniform_nurse(parent1, parent2)
            else:
                offspring = [row[:] for row in parent1]

            mutate(offspring, mutation_rate)
            
            next_generation.append(offspring)
            
        population = next_generation
        
        ranked_population = []
        for indiv in population:
            score = evaluate(indiv)
            ranked_population.append((score, indiv))
            
            if score < best_overall_score:
                best_overall_score = score
                best_overall_schedule = [row[:] for row in indiv]
        
        ranked_population.sort(key=lambda x: x[0])

    return best_overall_schedule


def tournament_selection(ranked_pop, k):
    candidates = random.sample(ranked_pop, k)
    best_candidate = min(candidates, key=lambda x: x[0])
    return best_candidate[1]


def crossover_uniform_nurse(p1, p2):

    offspring = []
    for i in range(len(p1)):
        if random.random() < 0.5:
            offspring.append(p1[i][:])
        else:
            offspring.append(p2[i][:])
    return offspring


def mutate(schedule, rate):

    for r in range(len(schedule)):
        for c in range(len(schedule[0])):
            if random.random() < rate:
                schedule[r][c] = random.choice([0, 1, 2])