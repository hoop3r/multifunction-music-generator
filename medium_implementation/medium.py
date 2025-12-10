""" 
I separated the medium project into the same format as BB so that it's a bit easier to digest

-medium -> main driver
-medium_genetic -> genetic algorithm functions 
-medium_synth -> synth config & pyo functions
-medium_logging -> debug functions

run medium.py to execute the program :)

general notes:
-Messed around with the fitness function to take user ratings into account
-Instead of opening a GUI window for each rating, it all exists within the command line.
The GUI will pop up for the final sequence only! 

-shortened the lead synth loop and messed with the sound settings ( I think it sounds neat! )


"""
import os
import random
import time
import numpy as np
from midiutil import MIDIFile
from pyo import *

from medium_synth import play_sequence_pyo, rate_sequence_cli, save_sequence_wav
from medium_genetic import runEvolution, flatten, POPULATION_SIZE, MAX_GENERATIONS, MUTATION_RATE, new_genome_id, generateGenome, compute_fitness, TEST_RNG
from medium_logging import dbg, error, set_debug

# Toggle this to enable debug printing across modules.
DEBUG = True
set_debug(DEBUG)

# Use the single canonical test RNG seed defined in `medium_genetic.py`.
test_rng = TEST_RNG

# MIDI information will be encoded by list of numbers corresponding to note codes
# Dictionary containing the patterns of tones and semitones that a given scale follows (this is for two octaves)
scaleStructures = {
    "major": [2, 2, 1, 2, 2, 2, 1] * 2,
    "minor": [2, 1, 2, 2, 1, 2, 2] * 2,
    "major pentatonic": [2, 2, 3, 2, 3] * 2,
    "minor pentatonic": [3, 2, 2, 3, 2] * 2,
}

# Dictionary containing the MIDI code for every note starting from A below middle C
scales = {}
a3 = 45
currentCode = a3
for i in "abcdefg":
    scales[i] = currentCode
    # B and E do not have sharps so skip them for this part
    if i != "b" and i != "e":
        scales[f"{i}#"] = currentCode + 1
        currentCode += 2
    # C and F do not have flats so skip them for this part
    if i != "c" and i != "f":
        scales[f"{i}b"] = currentCode - 1
        currentCode += 2
    else:
        currentCode += 1


def main():
    """Main function: read user preferences, choose algorithm, run it, and play result."""
    # Choose algorithm
    print("Choose algorithm:\n1) Genetic Algorithm (GA)\n2) Greedy baseline (hill-climb)")
    alg_choice = input("Select 1 or 2 (default 1): ").strip() or "1"
    use_greedy = False
    if alg_choice.startswith("2"):
        use_greedy = True

    print("Which scale?")

    scaleOptions = [scale for scale in scaleStructures.keys()]

    for i, option in enumerate(scaleOptions):
        print(f"{i+1}. {option.title()}?")
    key = input(" ").lower().strip()

    while key not in scaleStructures.keys():
        print("Invalid.")
        for i, option in enumerate(scaleOptions):
            print(f"{i+1}. {option.title()}?")
        key = input().lower().strip()

    root = input("Enter the root of your scale: ").lower().strip()
    while root not in scales.keys():
        root = input("Invalid. Enter the root of your scale: ").lower().strip()

    tempo = input("Pick a tempo (integer) between 30 and 300 bpm: ").strip()
    while not isValidTempo(tempo):
        tempo = input("Invalid. Pick a tempo (integer) between 30 and 300 bpm: ").strip()
    tempo = int(tempo)

    scale = buildScale(root, key)
    if use_greedy:
        # Run greedy baseline hill-climb; pass the shared integer seed
        res = greedy_baseline(scale, iterations=MAX_GENERATIONS, tempo=tempo, rng=test_rng)
    else:
        # Run evolution with interactive user rating via the CLI -> press Enter to stop playback and rate
        res = runEvolution(
            MUTATION_RATE,
            scale,
            user_rating_callback=rate_sequence_cli,
            playback_tempo=tempo,
            playback_rng=test_rng,
        )
    # Save the top result to WAV (and then open GUI playback for listening)
    try:
        # runEvolution returns (population, fitness_stats, pop_history)
        if isinstance(res, tuple):
            # support (population, fitness_stats, pop_history) or older single-list return
            if len(res) >= 3:
                population, fitness_stats, pop_history = res[0], res[1], res[2]
            elif len(res) == 2:
                population, fitness_stats = res
                pop_history = {"initial": None, "final": None}
            else:
                population = res[0]
                fitness_stats = None
                pop_history = {"initial": None, "final": None}
        else:
            population = res
            fitness_stats = None
            pop_history = {"initial": None, "final": None}

        # plot fitness statz
        if fitness_stats:
            try:
                from medium_plots import plot_from_run

                # Choose title and x-axis label depending on algorithm
                if use_greedy:
                    alg_name = 'greedy'
                    title = "Greedy Baseline Fitness History"
                else:
                    alg_name = 'ga'
                    title = "GA Fitness History"

                out = plot_from_run(fitness_stats, title=title, filename=None, show=False, algorithm=alg_name)
                dbg(f"Saved fitness history plot to {out}")
                print(f"Saved fitness history plot to {out}")
            except Exception as e:
                # Log the plotting failure 
                error(f"Failed to generate fitness plot: {e}")

        top_entry = population[0]
        top_notes = top_entry[1]
        fname = f"final_{int(time.time())}.wav"
        try:
            # pass the integer seed to save_sequence_wav so it can create a
            # deterministic RNG internally if needed
            save_sequence_wav(top_notes, tempo=tempo, filename=fname, duration=12.0, rng=test_rng)
            print(f"Saved final WAV to {fname}")
        except Exception as e:
            error(f"final WAV save failed: {e}")

        # Play the top result using pyo GUI for interactive listening
        try:
            # playback: pass the shared integer seed so playback routines can
            # instantiate deterministic RNGs consistently.
            play_sequence_pyo(top_notes, tempo, rng=test_rng, use_pyo_gui=True)
        except Exception as e:
            error(f"pyo GUI playback failed: {e}")
    except Exception as e:
        error(f"final playback/save failed: {e}")


def buildScale(root, key):
    """Builds scale based on passed in root and key by accessing pattern and starting note dictionaries."""
    rootCode = scales[root]
    scale = [rootCode]
    pattern = scaleStructures[key]
    currentCode = rootCode

    for j in pattern:
        currentCode += j
        scale.append(currentCode)
    return scale


def writeMidiToDisk(sequence, filename="out", userTempo=60):
    """Writes the generated sequence to a MIDI file."""
    time_pos = 0
    track = 0
    channel = 0
    tempo = userTempo
    volume = 100

    midiFile = MIDIFile(1)
    midiFile.addTempo(track, 0, tempo)
    fSequence = flatten(sequence)
    for pitch in fSequence:
        duration = random.choice([0.5, 0.75, 1])
        if pitch is not None:
            midiFile.addNote(track, channel, pitch, time_pos, duration, volume)
        time_pos += random.choice([0.25, 0.5, 1])


def isValidTempo(val):
    """Returns True if the value is a valid tempo (int between 30 and 300)."""
    try:
        val = int(val)
    except Exception:
        return False
    return 30 <= val <= 300


def greedy_baseline(scale: list, iterations: int = 100, tempo: int = 60, rng=None):
    """Simple greedy baseline"""
    if isinstance(rng, int):
        rng = random.Random(rng)
    elif isinstance(rng, random.Random):
        pass
    elif rng is None:
        # Default to shared TEST_RNG so callers don't need to pass rng everywhere.
        from medium_genetic import TEST_RNG
        rng = random.Random(TEST_RNG)
    else:
        raise TypeError(f"Unsupported rng type {type(rng)!r}. Provide an int seed or random.Random instance.")

    try:
        seed_val = None
        if hasattr(rng, "getrandbits"):
            seed_val = rng.getrandbits(64)
        elif isinstance(rng, random.Random):
            seed_val = TEST_RNG
        if seed_val is not None:
            random.seed(seed_val)
    except Exception:
        pass

    def _randrange(n):
        if hasattr(rng, "randrange"):
            return rng.randrange(n)
        if hasattr(rng, "integers"):
            return int(rng.integers(0, n))
        return random.randrange(n)

    def _choice(seq):
        if hasattr(rng, "choice"):
            return rng.choice(seq)
        return random.choice(seq)

    # initial genome
    gid = new_genome_id()
    genome = generateGenome(scale)
    current_fit = compute_fitness(genome)

    fitness_stats = {"min": [], "max": [], "mean": []}
    pop_history = {"initial": (gid, genome)}

    # record initial
    fitness_stats["min"].append(float(current_fit))
    fitness_stats["max"].append(float(current_fit))
    fitness_stats["mean"].append(float(current_fit))

    flat_len = len(flatten(genome))

    for it in range(iterations):
        # propose a candidate by mutating one random position
        cand = [list(bar) for bar in genome]
        # choose bar and step
        b = _randrange(len(cand))
        s = _randrange(len(cand[0]))
        cand[b][s] = _choice(scale)
        cand_fit = compute_fitness(cand)
        if cand_fit > current_fit:
            genome = cand
            current_fit = cand_fit

        # record stats (single-population -> min=max=mean=current)
        fitness_stats["min"].append(float(current_fit))
        fitness_stats["max"].append(float(current_fit))
        fitness_stats["mean"].append(float(current_fit))

    pop_history["final"] = (gid, genome)
    population = [(gid, genome)]
    return population, fitness_stats, pop_history


if __name__ == "__main__":
    main()
