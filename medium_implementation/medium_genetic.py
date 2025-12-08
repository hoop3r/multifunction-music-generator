import random
from typing import List, Tuple, Optional, Union

# Genetic algorithm hyperparameters
POPULATION_SIZE = 100
MAX_GENERATIONS = 100
MUTATION_RATE = 0.05
MAX_FITNESS = 30

# Lead loop size (bars × steps per bar)
LOOP_BARS = 2
STEPS_PER_BAR = 8

# User-rating # per gen
USER_SAMPLE_COUNT = 5

# User-rating multipliers
USER_MULTIPLIER_MAP = {5: 2.0, 4: 1.2, 3: 1.0, 2: 0.8, 1: 0.6, 0: 0.2}

USER_BONUS = 40

def flatten(arr: List[List[Optional[int]]]) -> list:
    """Flatten a 2D list of lists into a 1D list."""
    out = []
    for i in arr:
        if isinstance(i, list):
            out.extend(i)
        else:
            out.append(i)
    return out


def generateGenome(scale: list) -> List[List[int]]:
    """Generates one note sequence (LOOP_BARS × STEPS_PER_BAR)"""
    return [[random.choice(scale) for _ in range(STEPS_PER_BAR)] for _ in range(LOOP_BARS)]


def generatePopulation(n: int, scale: list) -> List[List[List[int]]]:
    """Generate n note sequences."""
    return [generateGenome(scale) for x in range(n)]


def compute_fitness(genome: List[List[Optional[int]]], user_rating: Optional[int] = None) -> float:
    """Compute a fitness for a genome"""
    smoothnessScore = 0.0
    rhythmScore = 0.0
    harmonyScore = 0.0

    harmonyIntervalsTable = {0: -20, 1: 5, 2: 5, 3: 50, 4: 50, 5: 30, 6: -10,
                             7: 50, 8: 10, 9: 40, 10: -2, 11: -2, 12: 10,
                             13: -5, 14: 5, 15: 5, 16: 50, 17: 50, 18: 30,
                             19: -10, 20: 50, 21: 10, 22: 40, 23: -2, 24: -2, 25: 10}

    flat = flatten(genome)
    numRests = flat.count(None)
    consecutiveRests = 0

    for i, note in enumerate(flat):
        if i != 0 and note is not None and flat[i - 1] is not None:
            prevNote = flat[i - 1]
            noteDifference = abs(note - prevNote)
            harmonyScore += harmonyIntervalsTable.get(noteDifference, 0)

            if noteDifference == 0:
                smoothnessScore -= 3
            elif noteDifference <= 2:
                smoothnessScore += 1
            elif noteDifference == 11:
                smoothnessScore -= 2
            else:
                smoothnessScore += 1 / noteDifference

            if abs(note - (prevNote + 12)) in (1, 2) or abs((note + 12) - prevNote) in (1, 2):
                smoothnessScore += 0.5

        if i != 0 and note is None and flat[i - 1] is None:
            consecutiveRests += 1

    if numRests * 10 <= len(flat):
        rhythmScore += 10

    penalty = 10
    if consecutiveRests:
        rhythmScore -= (consecutiveRests * penalty)

    # Aggregate to base fitness
    smoothnessWeight = 15
    rhythmWeight = 5
    harmonyWeight = 20
    base = (smoothnessScore * smoothnessWeight) + (rhythmScore * rhythmWeight) + (harmonyScore * harmonyWeight)

    # If no user rating provided return base fitness
    if user_rating is None:
        return float(base)

    # Apply user influence (multiplier + additive bonus)
    mult = USER_MULTIPLIER_MAP.get(user_rating, 1.0)
    bonus = USER_BONUS * (user_rating - 2.5)
    return float(base * mult + bonus)


def selectParents(population: list, ratings_map: dict = None) -> Tuple[list, list]:
    """Selects two sequences from the population. Probability of being selected is weighted by the fitness score of each sequence """
    fitnesses = []
    for idx, genome in enumerate(population):
        user_rating = None
        if ratings_map and idx in ratings_map:
            user_rating = ratings_map[idx]
        fitnesses.append(compute_fitness(genome, user_rating=user_rating))

    # Ensure non-negative weights
    min_f = min(fitnesses)
    if min_f < 0:
        fitnesses = [f - min_f for f in fitnesses]

    total = sum(fitnesses)
    if total == 0:
        parentA, parentB = random.sample(population, 2)
    else:
        parentA, parentB = random.choices(population, weights=fitnesses, k=2)
    return parentA, parentB


def crossoverFunction(parentA: list, parentB: list) -> Tuple[list, list]:
    """ Performs single point crossover on two sequences """
    noteStringA = flatten(parentA)
    noteStringB = flatten(parentB)

    if len(noteStringA) != len(noteStringB):
        raise ValueError("Parents must have equal flattened length for crossover")
    elif len(parentA) < 2:
        return parentA, parentB

    singlePoint = random.randint(1, len(noteStringA) - 1)
    childAFlat = noteStringA[:singlePoint] + noteStringB[singlePoint:]
    childBFlat = noteStringB[:singlePoint] + noteStringA[singlePoint:]

    childA = []
    childB = []
    barLength = len(parentA[0])
    sequenceLength = len(noteStringA)
    start = 0
    end = barLength
    while end <= sequenceLength:
        childA.append(childAFlat[start:end])
        childB.append(childBFlat[start:end])
        start = end
        end += barLength
    return childA, childB


def mutateGenome(genome: list, mutationRate: float, scale: list, rest_prob: float = 0.15) -> list:
    """Mutates notes and occasionally introduces rests (in-place)."""
    for i, bar in enumerate(genome):
        for j, note in enumerate(bar):
            if random.random() <= mutationRate:
                if random.random() < rest_prob:
                    genome[i][j] = None
                else:
                    genome[i][j] = random.choice(scale)
    return genome


def runEvolution(
    mutationRate: float,
    scale: list,
    user_rating_callback: Optional[callable] = None,
    user_sample_count: int = USER_SAMPLE_COUNT,
    rng: Optional[random.Random] = None,
    initial_rate_all: bool = False,
    playback_tempo: int = 60,
    playback_rng: Optional[object] = None,
) -> list:
    """ Runs genetic algorithm until a genome with the specified MAX_FITNESS score has been reached"""
    if rng is None:
        rng = random

    population = generatePopulation(POPULATION_SIZE, scale)

    # Ask the user to rate the entire initial population
    initial_ratings_map = {}
    if user_rating_callback is not None and initial_rate_all:
        for idx in range(len(population)):
            try:
                rating = user_rating_callback(population[idx], tempo=playback_tempo, rng=playback_rng)
            except Exception:
                rating = None
            if isinstance(rating, int) and 0 <= rating <= 5:
                initial_ratings_map[idx] = rating

    for gen in range(MAX_GENERATIONS):
        ratings_map = {}
        if gen == 0 and initial_ratings_map:
            ratings_map.update(initial_ratings_map)
        if user_rating_callback is not None and user_sample_count > 0:
            sample_count = min(user_sample_count, len(population))
            sampled_indices = rng.sample(range(len(population)), sample_count)
            for idx in sampled_indices:
                try:
                    rating = user_rating_callback(population[idx], tempo=playback_tempo, rng=playback_rng)
                except Exception:
                    rating = None
                if isinstance(rating, int) and 0 <= rating <= 5:
                    ratings_map[idx] = rating

        # Sort by base fitness for reporting/elitism
        population.sort(key=lambda g: compute_fitness(g), reverse=True)

        # Early stopping using base fitness
        best_fitness = compute_fitness(population[0])
        if best_fitness >= MAX_FITNESS:
            break

        nextGen = population[:2]  # elitism

        while len(nextGen) < POPULATION_SIZE:
            A, B = selectParents(population, ratings_map)
            childA, childB = crossoverFunction(A, B)

            childA = mutateGenome(childA, mutationRate, scale)
            childB = mutateGenome(childB, mutationRate, scale)

            nextGen.extend([childA, childB])

        population = nextGen[:POPULATION_SIZE]

    population.sort(key=lambda g: compute_fitness(g), reverse=True)
    return population
