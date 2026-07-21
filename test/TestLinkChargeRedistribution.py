"""Unit tests for Z1 / DZ1 link-atom charge redistribution.

Covers the new ``apply_link_charge_redistribution`` helper in
``openmmml.embeddings._links``. Does not exercise the full MACE stack —
that's covered by the existing electrostatic-embedding smoke tests.
"""
import numpy as np
import openmm as mm
import openmm.app as app
from openmm.app import element as elem
import pytest

from openmmml.embeddings._links import apply_link_charge_redistribution


# -----------------------------------------------------------------------------
# Helpers: build minimal Topology fixtures.
# -----------------------------------------------------------------------------

def _topology_with_bonds(bonds: list[tuple[int, int]], n_atoms: int = 4):
    """Tiny topology: a single chain with the requested bonds.

    All atoms are CARBON (element doesn't matter for these tests).
    """
    topo = app.Topology()
    chain = topo.addChain()
    residue = topo.addResidue("X", chain)
    atoms = [topo.addAtom(f"C{i}", elem.carbon, residue) for i in range(n_atoms)]
    for a, b in bonds:
        topo.addBond(atoms[a], atoms[b])
    return topo


def _toy_linkinfo(q_globals, m_globals):
    """Minimum link_info dict expected by apply_link_charge_redistribution."""
    return {
        "q_global": np.asarray(q_globals, dtype=np.int64),
        "m_global": np.asarray(m_globals, dtype=np.int64),
        "target_dist": np.full(len(q_globals), 1.09, dtype=np.float64),
    }


# -----------------------------------------------------------------------------
# Basic API
# -----------------------------------------------------------------------------

def test_none_returns_copy_unchanged():
    mm_atoms = np.array([0, 1, 2, 3], dtype=np.int64)
    mm_q = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64)
    out = apply_link_charge_redistribution(
        mm_atoms, mm_q, _toy_linkinfo([10], [1]),
        topology=_topology_with_bonds([]), scheme="none",
    )
    np.testing.assert_array_equal(out, mm_q)
    assert out is not mm_q  # must be a copy


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="Unsupported linkChargeScheme"):
        apply_link_charge_redistribution(
            np.array([0]), np.array([0.0]),
            _toy_linkinfo([], []),
            topology=_topology_with_bonds([]), scheme="z3",
        )


# -----------------------------------------------------------------------------
# Z1
# -----------------------------------------------------------------------------

def test_z1_zeros_M_only():
    mm_atoms = np.array([0, 1, 2, 3], dtype=np.int64)
    mm_q = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64)
    out = apply_link_charge_redistribution(
        mm_atoms, mm_q, _toy_linkinfo([10], [1]),
        topology=_topology_with_bonds([]), scheme="z1",
    )
    assert out[1] == 0.0
    np.testing.assert_array_equal(out[[0, 2, 3]], mm_q[[0, 2, 3]])


def test_z1_multiple_M_atoms():
    mm_atoms = np.array([0, 1, 2, 3], dtype=np.int64)
    mm_q = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64)
    out = apply_link_charge_redistribution(
        mm_atoms, mm_q, _toy_linkinfo([10, 11], [1, 3]),
        topology=_topology_with_bonds([]), scheme="z1",
    )
    assert out[1] == 0.0
    assert out[3] == 0.0
    np.testing.assert_array_equal(out[[0, 2]], mm_q[[0, 2]])


# -----------------------------------------------------------------------------
# DZ1
# -----------------------------------------------------------------------------

def test_dz1_distributes_to_MM_neighbors_and_preserves_total():
    # Bonds: M (atom 1) is bonded to MM atoms 2 and 3, plus Q atom 10.
    # The Q-M bond should be ignored (not an MM-MM bond).
    # q_M_orig = -0.2 -> each MM neighbour gets -0.1.
    topology = _topology_with_bonds([(1, 2), (1, 3)], n_atoms=4)
    mm_atoms = np.array([0, 1, 2, 3], dtype=np.int64)
    mm_q = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64)
    out = apply_link_charge_redistribution(
        mm_atoms, mm_q, _toy_linkinfo([10], [1]),
        topology=topology, scheme="dz1",
    )
    assert out[1] == 0.0
    assert out[2] == pytest.approx(0.3 + (-0.2) / 2)
    assert out[3] == pytest.approx(-0.4 + (-0.2) / 2)
    # Total preserved to round-off
    assert out.sum() == pytest.approx(mm_q.sum(), abs=1e-12)


def test_dz1_skips_Q_neighbors():
    # M is bonded to Q (10) AND to MM (2). Only the MM bond counts.
    # q_M_orig = -0.5 -> atom 2 gets +(-0.5)/1 = -0.5.
    topology = _topology_with_bonds([(1, 2), (1, 10)], n_atoms=11)
    mm_atoms = np.array([0, 1, 2, 3], dtype=np.int64)
    mm_q = np.array([0.1, -0.5, 0.2, 0.2], dtype=np.float64)
    out = apply_link_charge_redistribution(
        mm_atoms, mm_q, _toy_linkinfo([10], [1]),
        topology=topology, scheme="dz1",
    )
    assert out[1] == 0.0
    assert out[2] == pytest.approx(0.2 - 0.5)
    # atom 3 unchanged
    assert out[3] == pytest.approx(0.2)


def test_dz1_no_MM_neighbors_warns_and_falls_back_to_Z1():
    # M (atom 1) has no MM bonds — only bonded to Q (atom 10).
    topology = _topology_with_bonds([(1, 10)], n_atoms=11)
    mm_atoms = np.array([0, 1], dtype=np.int64)
    mm_q = np.array([0.5, -0.5], dtype=np.float64)
    with pytest.warns(UserWarning, match="no MM neighbours"):
        out = apply_link_charge_redistribution(
            mm_atoms, mm_q, _toy_linkinfo([10], [1]),
            topology=topology, scheme="dz1",
        )
    assert out[1] == 0.0
    # Total not preserved here — warning told us.
    assert out.sum() != pytest.approx(mm_q.sum(), abs=1e-12)


def test_dz1_multiple_M_atoms_each_get_own_neighbors():
    # M atoms = [1, 3]. Each is bonded to a different MM neighbour (0 and 2).
    # q_orig: atom 1 = -0.4 -> atom 0 += -0.4
    #         atom 3 = +0.6 -> atom 2 += +0.6
    topology = _topology_with_bonds([(1, 0), (3, 2)], n_atoms=4)
    mm_atoms = np.array([0, 1, 2, 3], dtype=np.int64)
    mm_q = np.array([0.0, -0.4, 0.0, 0.6], dtype=np.float64)
    out = apply_link_charge_redistribution(
        mm_atoms, mm_q, _toy_linkinfo([10, 11], [1, 3]),
        topology=topology, scheme="dz1",
    )
    assert out[1] == 0.0
    assert out[3] == 0.0
    assert out[0] == pytest.approx(-0.4)
    assert out[2] == pytest.approx(0.6)
    assert out.sum() == pytest.approx(mm_q.sum(), abs=1e-12)


def test_dz1_split_across_three_neighbors():
    # M (atom 1) has three MM neighbours (0, 2, 3). q_M_orig=-0.3 -> each +(-0.1).
    topology = _topology_with_bonds([(1, 0), (1, 2), (1, 3)], n_atoms=4)
    mm_atoms = np.array([0, 1, 2, 3], dtype=np.int64)
    mm_q = np.array([0.0, -0.3, 0.0, 0.0], dtype=np.float64)
    out = apply_link_charge_redistribution(
        mm_atoms, mm_q, _toy_linkinfo([10], [1]),
        topology=topology, scheme="dz1",
    )
    assert out[1] == 0.0
    for idx in (0, 2, 3):
        assert out[idx] == pytest.approx(-0.1)
    assert out.sum() == pytest.approx(mm_q.sum(), abs=1e-12)
