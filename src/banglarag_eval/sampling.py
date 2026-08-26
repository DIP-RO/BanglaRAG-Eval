"""Deterministic allocation of pilot examples to condition cells.

The pilot is structured as LANGUAGE CONDITION x EVIDENCE CONDITION. The
allocation procedure turns configured proportions into integer cell
counts deterministically:

1. The joint target for cell (language, evidence) is
   pilot_size * p_language * p_evidence  (independence assumption; the
   actual sampling strategy is documented in docs/pilot_design.md).
2. Each cell receives the floor of its target.
3. Remaining examples are assigned by largest fractional remainder,
   with ties broken by a seeded, reproducible shuffle of the stable
   cell ordering.

The same config always yields the same allocation, which is what makes
dataset versions reproducible. No perfectly balanced matrix is forced;
proportions are targets.
"""

from __future__ import annotations

import random

from banglarag_eval.config import PilotConfig


def allocate_cells(config: PilotConfig) -> dict[tuple[str, str], int]:
    """Compute integer example counts per (language, evidence) cell."""
    cells = [
        (language, evidence)
        for language in sorted(config.language_proportions)
        for evidence in sorted(config.evidence_proportions)
    ]

    targets = {
        (language, evidence): (
            config.pilot_size
            * config.language_proportions[language]
            * config.evidence_proportions[evidence]
        )
        for language, evidence in cells
    }

    allocation = {cell: int(targets[cell]) for cell in cells}
    remainder = config.pilot_size - sum(allocation.values())
    if remainder < 0 or remainder > len(cells):
        raise RuntimeError(
            f"largest-remainder invariant violated: remainder {remainder} "
            f"outside [0, {len(cells)}] for pilot_size {config.pilot_size}"
        )

    # Deterministic tie-breaking: stable sort by descending fractional
    # part; among equal fractions, order comes from a seeded shuffle of
    # the (already stable) cell list.
    rng = random.Random(config.random_seed)
    tie_break_order = list(cells)
    rng.shuffle(tie_break_order)
    tie_rank = {cell: rank for rank, cell in enumerate(tie_break_order)}

    by_fraction = sorted(
        cells,
        key=lambda cell: (-(targets[cell] - int(targets[cell])), tie_rank[cell]),
    )
    for cell in by_fraction[:remainder]:
        allocation[cell] += 1

    total = sum(allocation.values())
    if total != config.pilot_size:
        # Explicit raise (not assert): must survive python -O.
        raise RuntimeError(
            f"allocation total {total} != pilot_size {config.pilot_size}"
        )
    return allocation


def allocation_summary(allocation: dict[tuple[str, str], int]) -> dict[str, dict[str, int]]:
    """Marginal totals by language and by evidence condition."""
    by_language: dict[str, int] = {}
    by_evidence: dict[str, int] = {}
    for (language, evidence), count in allocation.items():
        by_language[language] = by_language.get(language, 0) + count
        by_evidence[evidence] = by_evidence.get(evidence, 0) + count
    return {"by_language": by_language, "by_evidence": by_evidence}
