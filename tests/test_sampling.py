"""Deterministic sampling allocation tests."""

from __future__ import annotations

from dataclasses import replace

from banglarag_eval.config import load_pilot_config
from banglarag_eval.constants import (
    EVIDENCE_CONDITIONS,
    LANGUAGE_CONDITIONS,
    REPO_ROOT,
)
from banglarag_eval.sampling import allocate_cells, allocation_summary

STAGE1 = REPO_ROOT / "configs" / "pilot_stage1.yaml"


def test_allocation_sums_to_pilot_size():
    config = load_pilot_config(STAGE1)
    allocation = allocate_cells(config)
    assert sum(allocation.values()) == config.pilot_size


def test_allocation_covers_full_condition_matrix():
    config = load_pilot_config(STAGE1)
    allocation = allocate_cells(config)
    assert set(allocation) == {
        (language, evidence)
        for language in LANGUAGE_CONDITIONS
        for evidence in EVIDENCE_CONDITIONS
    }


def test_allocation_is_deterministic():
    first = allocate_cells(load_pilot_config(STAGE1))
    second = allocate_cells(load_pilot_config(STAGE1))
    assert first == second


def test_allocation_scales_without_redesign():
    """The same procedure must handle 10 -> 50 -> 500 examples."""
    config = load_pilot_config(STAGE1)
    for size in (10, 30, 40, 50, 300, 400, 500):
        allocation = allocate_cells(replace(config, pilot_size=size))
        assert sum(allocation.values()) == size
        assert all(count >= 0 for count in allocation.values())


def test_marginals_track_configured_proportions():
    config = load_pilot_config(STAGE1)
    allocation = allocate_cells(config)
    summary = allocation_summary(allocation)
    # Largest-remainder over the joint grid keeps every cell within 1 of
    # its real-valued target, so each marginal can drift by at most the
    # number of cells contributing to it.
    for language, proportion in config.language_proportions.items():
        target = config.pilot_size * proportion
        assert abs(summary["by_language"][language] - target) <= len(
            EVIDENCE_CONDITIONS
        )
    for evidence, proportion in config.evidence_proportions.items():
        target = config.pilot_size * proportion
        assert abs(summary["by_evidence"][evidence] - target) <= len(
            LANGUAGE_CONDITIONS
        )


def test_same_seed_same_allocation_different_seed_may_differ():
    config = load_pilot_config(STAGE1)
    baseline = allocate_cells(config)
    assert allocate_cells(replace(config)) == baseline
    # A different seed changes only tie-breaking; totals stay intact.
    reseeded = allocate_cells(replace(config, random_seed=config.random_seed + 1))
    assert sum(reseeded.values()) == config.pilot_size
