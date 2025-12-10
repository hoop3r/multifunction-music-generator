from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _ensure_plots_dir() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    plots_dir = os.path.join(root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def plot_history(fitness_stats: Dict[str, List[float]], title: str, filename: Optional[str] = None, show: bool = False) -> str:
    """Plot min, max, and mean fitness over generations and save PNG."""
    
    # Normalize arrays
    mins = np.asarray(fitness_stats.get("min", []), dtype=float)
    maxs = np.asarray(fitness_stats.get("max", []), dtype=float)
    means = np.asarray(fitness_stats.get("mean", []), dtype=float)

    if mins.size == 0 and maxs.size == 0 and means.size == 0:
        raise ValueError("fitness_stats must contain at least one of 'min', 'max', or 'mean')")

    num_gen = np.arange(max(len(mins), len(maxs), len(means)))

    fig, ax = plt.subplots(1, figsize=(10, 7))

    if mins.size > 0:
        ax.plot(np.arange(len(mins)), mins, label="Min")
    if maxs.size > 0:
        ax.plot(np.arange(len(maxs)), maxs, label="Max")
    if means.size > 0:
        ax.plot(np.arange(len(means)), means, label="Mean")

    ax.legend()
    ax.set_xlabel("Generation", fontweight="bold")
    ax.set_ylabel("Fitness", fontweight="bold")
    ax.grid(axis="y")
    ax.set_title(title, fontweight="bold", fontsize=14)

    plots_dir = _ensure_plots_dir()
    if filename is None:
        safe_title = "_".join(title.strip().split())
        filename = f"ga_history_{safe_title}_{int(time.time())}.png"

    out_path = os.path.join(plots_dir, filename)
    fig.savefig(out_path, bbox_inches="tight")

    if show:
        try:
            plt.show()
        except Exception:
            pass

    plt.close(fig)
    return out_path


def plot_from_run(fitness_stats: Dict[str, List[float]], title: Optional[str] = None, filename: Optional[str] = None, show: bool = False) -> str:
    if title is None:
        title = "GA Fitness History"

    return plot_history(fitness_stats, title=title, filename=filename, show=show)
