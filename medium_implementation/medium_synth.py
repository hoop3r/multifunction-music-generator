# synthesizer and playback utilities for the music GA

import numpy as np
import time as _time
from typing import Optional, Tuple
from pyo import *
from medium_genetic import flatten
from medium_logging import dbg
import os


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
    return s, pat


def rate_sequence_cli(sequence, tempo: int = 60, rng: np.random.Generator = None) -> Optional[int]:
    """
    Play sequence without GUI, record while playing, let the user stop via Enter,
    then prompt for a 0-6 rating. If rating == 6, move the temp file to a saved super-like.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Start playback
    try:
        s_pat = play_sequence_pyo(sequence, tempo=tempo, rng=rng, use_pyo_gui=False)
        if not s_pat:
            return None
        s, pat = s_pat
    except Exception as e:
        dbg(f"[LEAD] playback failed: {e}")
        return None

    # Start recording immediately
    temp_fname = f"_temp_recording_{int(_time.time())}.wav"

    try:
        if hasattr(s, "recordOptions"):
            try:
                s.recordOptions(filename=temp_fname)
            except TypeError:
                s.recordOptions(temp_fname)
    except Exception as e:
        dbg(f"rate_sequence_cli: recordOptions failed: {e}")

    try:
        if hasattr(s, "recstart"):
            s.recstart()
        elif hasattr(s, "recStart"):
            s.recStart()
    except Exception as e:
        dbg(f"rate_sequence_cli: failed to start recording: {e}")

    print("Playing sequence (recording). Press Enter to stop and rate:")

    # Wait for user to hit Enter     
    try:
        input()
    except EOFError:
        try:
            pat.stop()
        except Exception:
            pass
        try:
            if hasattr(s, "recstop"):
                s.recstop()
            elif hasattr(s, "recStop"):
                s.recStop()
        except Exception:
            pass
        return None

    # Stop playback and recording
    try:
        pat.stop()
    except Exception:
        pass

    try:
        if hasattr(s, "recstop"):
            s.recstop()
        elif hasattr(s, "recStop"):
            s.recStop()
    except Exception as e:
        dbg(f"rate_sequence_cli: failed to stop recording cleanly: {e}")

    # Ask user for rating. If the response is anything but 6, discard the temp file.
    try:
        resp = input("Rate 0-6 (6 = Super like, blank to skip): ").strip()
    except EOFError:
        resp = ""

    # Blank input → delete temp and return None
    if resp == "":
        if os.path.exists(temp_fname):
            os.remove(temp_fname)
        return None

    # 6 → super-like: finalize temp file
    if resp == "6":
        final_fname = f"superlike_{int(_time.time())}.wav"
        try:
            os.rename(temp_fname, final_fname)
            print(f"Saved super-like WAV to {final_fname}")
        except Exception as e:
            print(f"Failed to finalize WAV: {e}")
        return 6

    # Any other response → remove temp file
    if os.path.exists(temp_fname):
        try:
            os.remove(temp_fname)
        except Exception:
            pass

    try:
        val = int(resp)
        return val
    except Exception:
        return None



def save_sequence_wav(sequence, tempo: int = 60, filename: Optional[str] = None, duration: float = 12.0, rng: np.random.Generator = None):
    """
    Save a sequence to WAV using pyo, by playing and recording in real time.
    """
    if filename is None:
        filename = f"export_{int(_time.time())}.wav"
    if rng is None:
        rng = np.random.default_rng()

    # Start playback
    s_pat = play_sequence_pyo(sequence, tempo=tempo, rng=rng, use_pyo_gui=False)
    if not s_pat:
        raise RuntimeError("play_sequence_pyo failed; cannot record final sequence")

    s, pat = s_pat

    # Setup recorder
    try:
        if hasattr(s, "recordOptions"):
            try:
                s.recordOptions(filename=filename)
            except TypeError:
                s.recordOptions(filename)
    except Exception as e:
        dbg(f"save_sequence_wav: recordOptions failed: {e}")

    # Start recording
    try:
        if hasattr(s, "recstart"):
            s.recstart()
        elif hasattr(s, "recStart"):
            s.recStart()
    except Exception as e:
        dbg(f"save_sequence_wav: recstart failed: {e}")
        try:
            pat.stop()
        except:
            pass
        raise

    # Play the pattern during recording
    try:
        pat.play()
    except Exception:
        pass

    # Sleep for the requested duration
    try:
        _time.sleep(duration)
    except:
        pass

    # Stop recording + playback
    try:
        if hasattr(s, "recstop"):
            s.recstop()
        elif hasattr(s, "recStop"):
            s.recStop()
    except Exception as e:
        dbg(f"save_sequence_wav: failed to stop recording: {e}")

    try:
        pat.stop()
    except:
        pass

    return filename
