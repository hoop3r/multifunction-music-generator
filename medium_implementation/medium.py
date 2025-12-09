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
from medium_genetic import runEvolution, flatten, POPULATION_SIZE, MAX_GENERATIONS, MUTATION_RATE
from medium_logging import dbg, error, set_debug

# Driver-wide debug flag. Toggle this to enable debug printing across modules.
DEBUG = True
set_debug(DEBUG)

test_rng = np.random.default_rng(192)

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
    """Main function: read user preferences, run GA, and play result."""
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
        top_entry = res[0]
        top_notes = top_entry[1]
        fname = f"final_{int(time.time())}.wav"
        try:
            save_sequence_wav(top_notes, tempo=tempo, filename=fname, duration=12.0, rng=test_rng)
            print(f"Saved final WAV to {fname}")
        except Exception as e:
            error(f"final WAV save failed: {e}")

        # Play the top result using pyo GUI for interactive listening (best-effort)
        try:
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


if __name__ == "__main__":
    main()
