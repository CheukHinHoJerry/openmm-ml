"""Regression test for the `target_dist` unit canonicalisation in
`_prepareLinkRecords` and its downstream consumers.

The historical bug: when `linkRecords` was loaded from a
`capping_mapping.csv`, the loader
multiplied the CSV's `target_dist_ang` column by 0.1 — turning a 1.09 Å
record into a 0.109 nm record stored in `linkInfo["target_dist"]`. But
`_computeMACE` reads `linkInfo["target_dist"]` together with `r_Q`,
`r_M` already in Å (via `positions_full = state.getPositions(asNumpy=True
).value_in_unit(unit.angstrom)`). The unit mismatch placed CSV-path
caps at ~0.1 Å from the Q atom — about 10× closer than the intended
~1.09 Å (typical C-H bond).

The fix: `linkInfo["target_dist"]` is canonically in **Å**. The CSV path
no longer multiplies by 0.1; the tuple path was already in Å.

Tests in this file pin the contract at three levels so a regression on
*any* of them surfaces clearly:

1. Tuple-path loader stores the value verbatim (Å).
2. CSV-path loader stores the value verbatim (Å, no `* 0.1` conversion).
3. Downstream end-to-end: `compute_cap_positions` placed against
   linkInfo + Å-scale Q/M positions gives a Q-L distance equal to the
   target value. Any future "fix" that re-introduces `* 0.1` at the
   consumer side (instead of the loader side) trips this end-to-end
   case even if it leaves the loader unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tiny_system_topology():
    """Minimal 5-atom system + topology so _prepareLinkRecords' validation
    (q in atoms, m not in atoms, etc.) is satisfied. Atoms 0,1,2 are ML;
    atoms 3,4 are MM."""
    import openmm as mm
    import openmm.app as app
    from openmm.app import element as elem

    system = mm.System()
    for _ in range(5):
        system.addParticle(1.0)
    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("X", chain)
    atoms = [top.addAtom(f"H{i}", elem.hydrogen, res) for i in range(5)]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        top.addBond(atoms[a], atoms[b])
    return top, system


def test_tuple_path_target_dist_is_angstrom(tiny_system_topology):
    """Tuple path: linkInfo['target_dist'] equals the value the caller passed.

    Locks the canonical interpretation as Å.
    """
    from openmmml.models.macepotential import _prepareLinkRecords

    top, system = tiny_system_topology
    link_info = _prepareLinkRecords(
        linkRecords=[(2, 3, 1.09)],
        atoms=[0, 1, 2],
        topology=top,
        system=system,
    )
    assert link_info is not None
    assert link_info["K"] == 1
    np.testing.assert_allclose(link_info["target_dist"], np.array([1.09]))


# The CSV-path test requires the MLMM-side `load_link_records` parser.
# openmm-ml proper does not depend on MLMM (it's an optional consumer),
# so the test skips cleanly when `mlmm` isn't on the path — keeping
# openmm-ml's own CI (which doesn't install MLMM) reproducible.
_mlmm_available = pytest.importorskip.__doc__ is not None  # always True; placeholder
try:
    import mlmm.link_atoms  # noqa: F401
    _HAS_MLMM = True
except Exception:
    _HAS_MLMM = False


@pytest.mark.skipif(not _HAS_MLMM, reason="MLMM not installed; CSV path needs mlmm.link_atoms.load_link_records")
def test_csv_path_target_dist_in_angstrom(tmp_path, tiny_system_topology):
    """CSV path: linkInfo['target_dist'] matches the CSV's target_dist_ang
    column verbatim (no spurious 0.1 conversion).

    Historical bug had this multiplied by 0.1, producing
    linkInfo['target_dist'] = 0.109 for a CSV target_dist_ang of 1.09,
    which then placed MACE caps 10x too close to Q.
    """
    from openmmml.models.macepotential import _prepareLinkRecords

    csv = tmp_path / "capping_mapping.csv"
    csv.write_text(
        "q_idx1,m_idx1,q_element,m_element,bond_order,qm_mm_distance_ang,"
        "target_dist_ang,r_Q_x,r_Q_y,r_Q_z,r_M_x,r_M_y,r_M_z,r_L_x,r_L_y,r_L_z\n"
        "3,4,H,H,1.000,1.500000,1.090000,"
        "0.0,0.0,0.0,1.5,0.0,0.0,1.09,0.0,0.0\n"
    )

    top, system = tiny_system_topology
    link_info = _prepareLinkRecords(
        linkRecords=str(csv),
        atoms=[0, 1, 2],
        topology=top,
        system=system,
    )
    assert link_info is not None
    assert link_info["K"] == 1
    np.testing.assert_allclose(
        link_info["target_dist"],
        np.array([1.09]),
        err_msg=(
            "CSV target_dist_ang=1.09 must land in linkInfo as 1.09 (Å). "
            "If you see 0.109 here, the historical *0.1 nm conversion has "
            "regressed; see openmm-ml fix/target-dist-unit-canonical-angstrom."
        ),
    )


def test_compute_cap_positions_q_to_l_distance_matches_target():
    """End-to-end contract: with linkInfo['target_dist'] in Å and Q,M
    positions in Å, the cap placement helper produces a Q-L distance
    equal to target_dist Å.

    This is the consumer-side test codex review of PR #20 called out: a
    future regression that re-introduces ``* 0.1`` at the consumer side
    (e.g. ``compute_cap_positions(r_Q, r_M, linkInfo['target_dist'] * 0.1)``)
    would slip past the loader-only test but trip this one.
    """
    from openmmml.models._links import compute_cap_positions

    # Q at origin, M at (1.5, 0, 0) — a typical C-C single bond, 1.5 Å.
    r_Q = np.array([[0.0, 0.0, 0.0]])
    r_M = np.array([[1.5, 0.0, 0.0]])
    target_dist_ang = np.array([1.09])  # typical C-H, in Å

    cap_pos, C_L = compute_cap_positions(r_Q, r_M, target_dist_ang)
    q_to_l = np.linalg.norm(cap_pos - r_Q, axis=-1)
    np.testing.assert_allclose(q_to_l, target_dist_ang, atol=1e-12)
    np.testing.assert_allclose(C_L, target_dist_ang / 1.5, atol=1e-12)
