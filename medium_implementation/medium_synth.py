# synthesizer and playback utilities for the music GA
"""synthesizer and playback utilities for the music GA

This module provides a small pyo-backed player (when pyo is installed) and a
CLI rating helper used by the genetic driver. The code is defensive so tests
or CI environments without pyo can still import and run non-audio parts.
"""


import numpy as np
from typing import Optional, Tuple

try:
    from pyo import Server, Adsr, SuperSaw, ButHP, Pattern
    HAVE_PYO = True
except Exception:
    HAVE_PYO = False

from medium_genetic import flatten
from medium_logging import dbg



def midi_to_freq(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))


def play_sequence_pyo(sequence, tempo: int = 60, rng: np.random.Generator = None, use_pyo_gui: bool = False) -> Optional[Tuple[object, object]]:
    """Play a sequence using pyo. Returns (server, pattern) in non-GUI mode so the caller can stop playback.
    If pyo is not available, returns None.
    """
    flat = flatten(sequence)
    if len(flat) == 0:
        dbg("Empty sequence; nothing to play.")
        return None

    if rng is None:
        rng = np.random.default_rng()

    durations = rng.choice([0.5, 0.75, 1, 2], size=len(flat)).tolist()
    intervals = rng.choice([0.25, 0.5, 1], size=len(flat)).tolist()

    beat = 60.0 / float(tempo)

    if not HAVE_PYO:
        dbg("pyo not available: skipping audio playback (falling back to MIDI/file-only behavior)")
        return None

    s = Server(sr=48000, duplex=0).boot()

    idx = {"i": 0}
    voice = {"obj": None}

    def step():
        i = idx["i"]

        if i >= len(flat):
            idx["i"] = 0
            i = 0

        pitch = flat[i]
        dur = durations[i] * beat
        interval = intervals[i] * beat

        dbg(f"STEP {i}: pitch={pitch}, dur={dur:.3f}, interval={interval:.3f}")

        if voice["obj"] is not None:
            try:
                voice["obj"].stop()
            except Exception:
                pass

        if pitch is not None:
            freq = midi_to_freq(pitch)
            env = Adsr(attack=0.01, decay=0.08, sustain=0.4, release=0.08, dur=dur, mul=0.2).play()
            osc = SuperSaw(freq=freq, detune=0.3, bal=0.5, mul=env)
            osc = ButHP(osc, freq=200)
            osc.out()
            voice["obj"] = osc
        else:
            voice["obj"] = None

        pat.time = float(interval)
        idx["i"] += 1

    pat = Pattern(step, time=float(intervals[0] * beat))

    s.start()
    dbg("[LEAD] server started; starting lead pattern")

    dbg("[LEAD] playing lead pattern")
    pat.play()
    if use_pyo_gui:
        dbg("[LEAD] entering pyo GUI (interactive). If running headless set GUI to False")
        try:
            s.gui(locals())
        except Exception:
            pass
        return None
    # Non-GUI mode: return server and pattern so caller can stop playback.
    return s, pat


def rate_sequence_cli(sequence, tempo: int = 60, rng: np.random.Generator = None) -> Optional[int]:
    """Play sequence without GUI, let the user stop via Enter, then prompt for a 0-5 rating.
       Returns an int 0-5 or None if skipped.
    """
    try:
        s_pat = play_sequence_pyo(sequence, tempo=tempo, rng=rng, use_pyo_gui=False)
        if not s_pat:
            return None
        s, pat = s_pat
    except Exception as e:
        dbg(f"[LEAD] playback failed: {e}")
        return None

    print("Playing sequence. Press Enter to stop playback and rate (blank to skip).")
    try:
        input()
    except EOFError:
        # Non-interactive
        try:
            pat.stop()
            s.stop()
        except Exception:
            pass
        return None

    try:
        pat.stop()
    except Exception:
        pass
    try:
        s.stop()
    except Exception:
        pass

    # Prompt for rating
    try:
        while True:
            resp = input("Rate 0-5 (blank to skip): ").strip()
            if resp == "":
                return None
            try:
                val = int(resp)
            except Exception:
                print("Please enter an integer between 0 and 5, or blank to skip.")
                continue
            if 0 <= val <= 5:
                return val
            else:
                print("Please enter an integer between 0 and 5, or blank to skip.")

    except EOFError:
        return None
