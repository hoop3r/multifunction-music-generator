"""
 crowding implementation for the GA.

"""

from typing import Any, Dict, List, Tuple


def flatten(arr: List[List[Any]]) -> list:
    """resolve circular import with this."""
    out = []
    for i in arr:
        if isinstance(i, list):
            out.extend(i)
        else:
            out.append(i)
    return out


def ham_dist(a: List[Any], b: List[Any]) -> int:
    """Hamming-like distance for two flattened genomes."""
    fa = flatten(a) if not isinstance(a, list) or any(isinstance(x, list) for x in a) else a
    fb = flatten(b) if not isinstance(b, list) or any(isinstance(x, list) for x in b) else b

    # Ensure list format
    la = list(fa)
    lb = list(fb)

    common = min(len(la), len(lb))
    dist = sum(1 for i in range(common) if la[i] != lb[i])
    dist += abs(len(la) - len(lb))
    return dist


def get_genotype(population: Any, key: Any) -> Any:
    """Retrieve a genotype from population given a key"""
    if isinstance(population, dict):
        return population[key]

    if isinstance(population, list):
        item = population[key]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            return item[1]
        return item

    raise TypeError("Unsupported population type for genotype lookup")


def crowding(population: Any, fitness: Any, replacements: Dict[Tuple[Any, Any], List[Any]], child_fitness: Dict[Tuple[Any, Any], List[float]]):
    """Implements deterministic crowding post reproduction"""
    new_replacements: Dict[Tuple[Any, Any], List[Any]] = {}
    new_child_fitness: Dict[Tuple[Any, Any], List[float]] = {}

    for pair in replacements.keys():
        p1, p2 = pair

        p1_ge = get_genotype(population, p1)
        p2_ge = get_genotype(population, p2)

        c1_ge = replacements[pair][0]
        c2_ge = replacements[pair][1]

        # Resolve fitness values
        try:
            p1_fit = fitness[p1]
        except Exception:
            p1_fit = fitness[p1]

        try:
            p2_fit = fitness[p2]
        except Exception:
            p2_fit = fitness[p2]

        c1_fit = child_fitness[pair][0]
        c2_fit = child_fitness[pair][1]

        # Start with the original parents as the default survivors
        new_replacements[pair] = [p1_ge, p2_ge]
        new_child_fitness[pair] = [p1_fit, p2_fit]

        # Determine pairing by distance
        left_dist = ham_dist(p1_ge, c1_ge) + ham_dist(p2_ge, c2_ge)
        right_dist = ham_dist(p1_ge, c2_ge) + ham_dist(p2_ge, c1_ge)

        if left_dist <= right_dist:
            # p1 closer to c1, p2 closer to c2
            if c1_fit > p1_fit:
                new_replacements[pair][0] = c1_ge
                new_child_fitness[pair][0] = c1_fit

            if c2_fit > p2_fit:
                new_replacements[pair][1] = c2_ge
                new_child_fitness[pair][1] = c2_fit
        else:
            # p2 closer to c1, p1 closer to c2
            if c1_fit > p2_fit:
                new_replacements[pair][1] = c1_ge
                new_child_fitness[pair][1] = c1_fit

            if c2_fit > p1_fit:
                new_replacements[pair][0] = c2_ge
                new_child_fitness[pair][0] = c2_fit

    return new_replacements, new_child_fitness