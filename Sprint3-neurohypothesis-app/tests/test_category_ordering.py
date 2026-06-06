"""
Tests for v2.1 category ordering (alphabetical complement after primary).

Verifies:
- order_categories returns [primary, comp_a, comp_b, ...] with comps alphabetical
- All 6 categories appear exactly once when primary is any of them
- select_categories returns (primary, []) for H1, (primary, [comp]) for H2..H6
"""

from __future__ import annotations

import pytest

import config


@pytest.mark.parametrize("primary", config.CATEGORIES)
def test_order_categories_starts_with_primary(primary: str) -> None:
    from src.engine.generate import order_categories
    ordered = order_categories(primary)
    assert ordered[0] == primary
    assert len(ordered) == len(config.CATEGORIES)
    assert set(ordered) == set(config.CATEGORIES)


@pytest.mark.parametrize("primary", config.CATEGORIES)
def test_complements_are_alphabetical(primary: str) -> None:
    from src.engine.generate import order_categories
    ordered = order_categories(primary)
    complements = ordered[1:]
    assert complements == sorted(complements), (
        f"complements not alphabetical for primary='{primary}': {complements}"
    )


def test_h1_uses_primary_only() -> None:
    from src.engine.generate import order_categories, select_categories
    primary = "Behavioral & Cognitive Neuroscience"
    ordered = order_categories(primary)
    p, comp = select_categories(ordered, hyp_index=0)
    assert p == primary
    assert comp == []


def test_h2_uses_first_alphabetical_complement() -> None:
    """For primary=Behavioral, H2 should use Animal Models (alphabetically first
    of the remaining 5)."""
    from src.engine.generate import order_categories, select_categories
    primary = "Behavioral & Cognitive Neuroscience"
    ordered = order_categories(primary)
    p, comp = select_categories(ordered, hyp_index=1)
    assert p == primary
    assert comp == ["Animal Models"]


def test_h6_uses_last_alphabetical_complement() -> None:
    """For primary=Animal Models, H6 should use Postmortem (alphabetically last)."""
    from src.engine.generate import order_categories, select_categories
    primary = "Animal Models"
    ordered = order_categories(primary)
    p, comp = select_categories(ordered, hyp_index=5)
    assert p == primary
    assert comp == ["Postmortem & Ex-Vivo Histology"]


def test_full_h1_through_h6_with_human_primary() -> None:
    """If primary = Human Neuroimaging, the complements (alphabetical, skipping H) are:
    Animal, Behavioral, Computational, Genetics, Postmortem."""
    from src.engine.generate import order_categories, select_categories
    primary = "Human Neuroimaging"
    ordered = order_categories(primary)

    expected = [
        (0, primary, []),
        (1, primary, ["Animal Models"]),
        (2, primary, ["Behavioral & Cognitive Neuroscience"]),
        (3, primary, ["Computational & Theoretical"]),
        (4, primary, ["Genetics & Molecular Biology"]),
        (5, primary, ["Postmortem & Ex-Vivo Histology"]),
    ]
    for idx, exp_p, exp_comp in expected:
        p, comp = select_categories(ordered, hyp_index=idx)
        assert p == exp_p, f"H{idx+1}: primary mismatch"
        assert comp == exp_comp, f"H{idx+1}: complement mismatch — got {comp}"
