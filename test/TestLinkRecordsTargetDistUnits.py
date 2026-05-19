"""Regression test for the `target_dist` unit canonicalisation in
`_prepareLinkRecords`.

The historical bug (R4 in the MLMM `TODO/oniom_parity_pr19_review.md`):
when `linkRecords` was loaded from a `capping_mapping.csv`, the loader
multiplied the CSV's `target_dist_ang` column by 0.1 — turning a 1.09 Å
record into a 0.109 nm record stored in `linkInfo["target_dist"]`. But
`_computeMACE` reads `linkInfo["target_dist"]` together with `r_Q`,
`r_M` already in Å (via `positions_full = state.getPositions(asNumpy=True
).value_in_unit(unit.angstrom)`). The unit mismatch placed CSV-path
caps at ~0.1 Å from the Q atom — about 10× closer than the intended
~1.09 Å (typical C-H bond).

The fix: `linkInfo["target_dist"]` is canonically in **Å**. The CSV path
no longer multiplies by 0.1; the tuple path was already in Å. The
oniom-electrostatic closure's `cap_info["target_dist"]` is converted to
nm by `_oniom_normalize_caps` (existing code, unchanged) so closure
positions stay in nm.

This file: directly probe `_prepareLinkRecords` for both paths and
assert the stored value matches the CSV column value (in Å).
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


def test_csv_path_target_dist_in_angstrom(tmp_path, tiny_system_topology):
    """CSV path: linkInfo['target_dist'] matches the CSV's target_dist_ang
    column verbatim (no spurious 0.1 conversion).

    Historical bug had this multiplied by 0.1, producing
    linkInfo['target_dist'] = 0.109 for a CSV target_dist_ang of 1.09,
    which then placed MACE caps 10x too close to Q.
    """
    from openmmml.models.macepotential import _prepareLinkRecords

    csv = tmp_path / "capping_mapping.csv"
    # Header matches the schema cli/cap_qm_boundary.py emits and that
    # mlmm.link_atoms.load_link_records parses.
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
