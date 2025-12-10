import random
import copy
from typing import List, Tuple, Optional, Union
import numpy as np

from medium_crowding import crowding
from medium_logging import dbg

# Genetic algorithm hyperparameters
POPULATION_SIZE = 100
MAX_GENERATIONS = 200
MUTATION_RATE = 0.07
MAX_FITNESS = 16000

# Lead loop size (bars × steps per bar)
LOOP_BARS = 2
STEPS_PER_BAR = 8

# User-rating # per gen
USER_SAMPLE_COUNT = 0

human_bonus_map = {}  # maps genome_id -> bonus float
HUMAN_BONUS_WEIGHT = 200.0

GLOBAL_GENOME_ID = 0

def new_genome_id():
    global GLOBAL_GENOME_ID
    GLOBAL_GENOME_ID += 1
    return GLOBAL_GENOME_ID

def flatten(arr: List[List[Optional[int]]]) -> list:
    """Flatten a 2D list/array of lists into a 1D list.

    This treats Python lists, tuples and numpy arrays as sequences to be
    flattened. It avoids treating strings/bytes as sequences.
    """
    out = []
    for i in arr:
        # treat common sequence types (list/tuple/numpy array) as iterable bars
        if isinstance(i, (list, tuple)) or (hasattr(i, 'shape') and hasattr(i, 'tolist')):
            try:
                out.extend(list(i))
            except Exception:
                out.append(i)
        else:
            out.append(i)
    return out


def generateGenome(scale: list) -> List[List[int]]:
    """Generates one note sequence (LOOP_BARS × STEPS_PER_BAR)"""
    return [[random.choice(scale) for _ in range(STEPS_PER_BAR)] for _ in range(LOOP_BARS)]


def generatePopulation(n: int, scale: list) -> List[List[List[int]]]:
    """Generate n note sequences."""
    return [(new_genome_id(), generateGenome(scale)) for _ in range(n)]

def compute_fitness(genome: List[List[Optional[int]]], human_bonus = 0.0) -> float:
    """Compute a fitness for a genome"""
    smoothnessScore = 0.0
    rhythmScore = 0.0
    harmonyScore = 0.0

    harmonyIntervalsTable = {0: 10, 1: 5, 2: 5, 3: 50, 4: 50, 5: 30, 6: -10,
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

    base += HUMAN_BONUS_WEIGHT * human_bonus
    return float(base)

def selectParents(population: list) -> Tuple[list, list]:
    """Selects two sequences from the population. Probability of being selected is weighted by the fitness score of each sequence """
    fitnesses = []
    for idx, genome in enumerate(population):
        gid, notes = genome
        bonus = human_bonus_map.get(gid, 0.0)
        fval = compute_fitness(notes, human_bonus=bonus)
        # Defensive: ensure fitness is a scalar float
        try:
            fval = float(fval)
        except Exception:
            raise TypeError(f"Computed fitness for population index {idx} (gid={gid}) is not numeric: {fval!r}")
        fitnesses.append(fval)

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
    return (new_genome_id(), childA), (new_genome_id(), childB)


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
    """Runs the genetic algorithm until MAX_FITNESS is reached.

    This function always collects fitness statistics per generation and stores
    an initial/final population snapshot in `pop_history`. It returns a tuple
    (population, fitness_stats, pop_history).
    """
    if rng is None:
        rng = random

    population = generatePopulation(POPULATION_SIZE, scale)

    # Ask the user to rate the entire initial population
    initial_ratings_map = {}
    if user_rating_callback is not None and initial_rate_all:
        for idx in range(len(population)):
            gid, notes = population[idx]
            try:
                # pass only the notes to the callback (not the whole tuple)
                rating = user_rating_callback(notes, tempo=playback_tempo, rng=playback_rng)
            except Exception:
                rating = None
            if isinstance(rating, int) and 0 <= rating <= 5:
                human_bonus = (rating / 5.0) * 2 - 1  # map 0–5 → -1…+1
                initial_ratings_map[idx] = human_bonus
                gid, _ = population[idx]
                human_bonus_map[gid] = human_bonus

    # collect fitness statistics and population history
    fitness_stats = {"min": [], "max": [], "mean": []}
    pop_history = {"initial": copy.deepcopy(population)}

    # Record initial fitness stats
    initial_fitness_vals = [compute_fitness(g[1], human_bonus_map.get(g[0], 0.0)) for g in population]
    if initial_fitness_vals:
        fitness_stats["min"].append(float(min(initial_fitness_vals)))
        fitness_stats["max"].append(float(max(initial_fitness_vals)))
        fitness_stats["mean"].append(float(sum(initial_fitness_vals) / len(initial_fitness_vals)))
    # debug logging for initial fitness
    try:
        dbg(f"[runEvolution] Initial fitness -> min={fitness_stats['min'][-1]:.2f}, max={fitness_stats['max'][-1]:.2f}, mean={fitness_stats['mean'][-1]:.2f}")
    except Exception:
        pass

    for gen in range(MAX_GENERATIONS):
        ratings_map = {}
        if gen == 0 and initial_ratings_map:
            ratings_map.update(initial_ratings_map)
        # Ask for user ratings only every 5 generations
        if user_rating_callback is not None and user_sample_count > 0 and gen % 5 == 0:
            # Print generation and current best fitness
            gid, notes = max(population, key=lambda g: compute_fitness(g[1], human_bonus_map.get(g[0], 0.0)))
            best_fitness = compute_fitness(notes, human_bonus_map.get(gid, 0.0))
            print(f"[Generation {gen}] Current max fitness: {best_fitness:.2f}")

            sample_count = min(user_sample_count, len(population))
            sampled_indices = rng.sample(range(len(population)), sample_count)

            # Ensure top 10% individual is included
            top_10_percent = int(0.1 * len(population))
            top_indices = np.argsort([compute_fitness(g[1], human_bonus_map.get(g[0], 0.0)) for g in population])[::-1][
                          :top_10_percent]

            # Pick one from top_indices to guarantee inclusion
            top_idx = rng.choice(top_indices)
            if top_idx not in sampled_indices:
                sampled_indices[0] = top_idx  # Replace the first sampled index

            for idx in sampled_indices:
                try:
                    gid, notes = population[idx]
                    rating = user_rating_callback(notes, tempo=playback_tempo, rng=playback_rng)
                except Exception:
                    rating = None
                if isinstance(rating, int) and 0 <= rating <= 5:
                    human_bonus = (rating / 5.0) * 2 - 1  # map 0–5 → -1…+1
                    ratings_map[idx] = human_bonus
                    gid, _ = population[idx]
                    human_bonus_map[gid] = human_bonus

        # Sort by base fitness for reporting/elitism
        population.sort(key=lambda g: compute_fitness(g[1], human_bonus_map.get(g[0], 0.0)),
                        reverse=True)

        # record stats for this generation
        fitness_vals = [compute_fitness(g[1], human_bonus_map.get(g[0], 0.0)) for g in population]
        if fitness_vals:
            fitness_stats["min"].append(float(min(fitness_vals)))
            fitness_stats["max"].append(float(max(fitness_vals)))
            fitness_stats["mean"].append(float(sum(fitness_vals) / len(fitness_vals)))
        # per-generation debug logging
        try:
            dbg(f"[runEvolution] Generation {gen} fitness -> min={fitness_stats['min'][-1]:.2f}, max={fitness_stats['max'][-1]:.2f}, mean={fitness_stats['mean'][-1]:.2f}")
        except Exception:
            pass

        # Early stopping using base fitness
        gid, notes = population[0]
        best_fitness = compute_fitness(notes, human_bonus_map.get(gid, 0.0))
        if best_fitness >= MAX_FITNESS:
            print(f"\nMax fitness ({MAX_FITNESS}) reached at generation {gen}")
            try:
                from medium_synth import play_sequence_pyo
                # Unpack the tuple to get just the genome notes
                best_id, best_notes = population[0]
                play_sequence_pyo(best_notes, tempo=playback_tempo, rng=playback_rng, use_pyo_gui=True)
            except Exception as e:
                print(f"Playback failed: {e}")
            break

        # Use deterministic crowding during reproduction.
        nextGen = population[:2]  # elitism (keep top 2)

        # Precompute parent fitnesses by index
        fitness_by_index = []
        for idx, (gid, notes) in enumerate(population):
            fitness_by_index.append(compute_fitness(notes, human_bonus_map.get(gid, 0.0)))

        # Helper to find parent's index by gid
        def _find_index_by_gid(pop, gid):
            for i, item in enumerate(pop):
                if isinstance(item, (tuple, list)) and len(item) >= 1 and item[0] == gid:
                    return i
            # fallback: random index
            return random.randrange(len(pop))

        replacements = {}
        child_fitness = {}
        child_id_map = {}

        # Number of pairings to create (each pairing produces two children)
        pairings = (POPULATION_SIZE - len(nextGen)) // 2
        for _ in range(pairings):
            A, B = selectParents(population)
            idA, notesA = A
            idB, notesB = B

            idxA = _find_index_by_gid(population, idA)
            idxB = _find_index_by_gid(population, idB)

            # crossover expects plain note-lists
            childA, childB = crossoverFunction(notesA, notesB)
            childA_id, childA_notes = childA
            childB_id, childB_notes = childB

            # inheritance: average parent bonuses
            parent_bonus = 0.5 * (human_bonus_map.get(idA, 0.0) + human_bonus_map.get(idB, 0.0))
            human_bonus_map[childA_id] = parent_bonus
            human_bonus_map[childB_id] = parent_bonus

            # mutate the child's notes
            childA_notes = mutateGenome(childA_notes, mutationRate, scale)
            childB_notes = mutateGenome(childB_notes, mutationRate, scale)

            # compute child fitness
            c1_fit = compute_fitness(childA_notes, human_bonus_map.get(childA_id, 0.0))
            c2_fit = compute_fitness(childB_notes, human_bonus_map.get(childB_id, 0.0))

            # Record replacements keyed by the parent indices
            replacements[(idxA, idxB)] = [childA_notes, childB_notes]
            child_fitness[(idxA, idxB)] = [c1_fit, c2_fit]
            child_id_map[(idxA, idxB)] = [childA_id, childB_id]

        # Apply deterministic crowding to decide survivors for each parent pair
        try:
            new_replacements, new_child_fitness = crowding(population, fitness_by_index, replacements, child_fitness)
        except Exception:
            # Fallback: if crowding fails, just accept children directly
            new_replacements = replacements
            new_child_fitness = child_fitness

        # Build next generation from survivors
        for pair, survivors in new_replacements.items():
            p1_idx, p2_idx = pair
            parent1_id, parent1_gen = population[p1_idx]
            parent2_id, parent2_gen = population[p2_idx]

            cA_notes, cB_notes = replacements.get(pair, [None, None])
            cA_id, cB_id = child_id_map.get(pair, [None, None])

            # For each survivor slot, determine whether it's childA/childB or parent1/parent2
            for slot_index in (0, 1):
                surv_gen = survivors[slot_index]
                if cA_notes is not None and surv_gen == cA_notes:
                    sid = cA_id
                    sgen = cA_notes
                elif cB_notes is not None and surv_gen == cB_notes:
                    sid = cB_id
                    sgen = cB_notes
                elif surv_gen == parent1_gen:
                    sid = parent1_id
                    sgen = parent1_gen
                elif surv_gen == parent2_gen:
                    sid = parent2_id
                    sgen = parent2_gen
                else:
                    # unexpected: fall back to parent1
                    sid = parent1_id
                    sgen = parent1_gen

                nextGen.append((sid, sgen))

        # Ensure population size
        population = nextGen[:POPULATION_SIZE]

    population.sort(
        key=lambda g: compute_fitness(g[1], human_bonus_map.get(g[0], 0.0)),
        reverse=True
    )

    # Log final best fitness
    try:
        if population:
            best_gid, best_notes = population[0]
            best_fit = compute_fitness(best_notes, human_bonus_map.get(best_gid, 0.0))
            dbg(f"[runEvolution] Final best fitness -> {best_fit:.2f} (gid={best_gid})")
    except Exception:
        pass

    # Store final population snapshot
    pop_history["final"] = copy.deepcopy(population)

    # Always return the population along with collected fitness statistics and
    # the population history dictionary.
    return population, fitness_stats, pop_history
