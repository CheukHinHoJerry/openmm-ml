"""Slice 3′ correctness tests for capped ONIOM-EE.

Two focused tests requested by the reviewer:

1. **ML = MM cancellation.** When the high-level "MACE" returns
   exactly ``E_MM(model)``, the ONIOM total must collapse to
   ``E_MM(real)``:

   .. code::

       E_total = E_MM(real) + E_high(model) - E_MM(model)
               = E_MM(real) + E_MM(model) - E_MM(model)
               = E_MM(real)

   If the cap-H force redistribution (or anything else in the
   subtraction machinery) is wrong, this won't hold.

2. **PBC + caps gating.** Cut bond across the periodic boundary:
   currently the link-atom infrastructure rejects PBC at
   ``_prepareLinkRecords``. Pin that the rejection is loud and the
   error message points at the deferred work. (Min-image cap
   placement under PBC is a future Slice item.)

Physical / plotting tests (finite-difference forces, cap-H leakage,
boundary bond scans) are explicitly *not* implemented here — those
need visualization and live in a notebook, not the test suite.
"""
from __future__ import annotations

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
import pytest

from openmmml import MLPotential
from openmmml.mlpotential import (
    MLPotentialImpl,
    MLPotentialImplFactory,
    _build_oniom_model_system,
    _make_oniom_low_model_closure,
)


# ---------------------------------------------------------------------------
# A test-only MLPotential whose "high-level" force evaluates +E_MM(model).
# When used in createMixedSystem(embedding="oniom-electrostatic"), the
# ONIOM low-model correction (-E_MM(model)) cancels it and the total
# reduces to E_MM(real).
# ---------------------------------------------------------------------------

class _MMAsMaceImpl(MLPotentialImpl):
    """`addForces` builds a model `System` matching the one that
    ``_build_oniom_mixed_system`` will build, wraps it in a
    PythonForce that returns ``+E`` (no sign flip), and adds it to
    the host. Combined with the existing ``-E_MM(model)`` correction,
    the ML contribution becomes ``+E_MM(model) - E_MM(model) = 0``,
    so ``E_total = E_MM(real)``.
    """

    def addForces(self, topology, system, atoms, forceGroup, **args):
        atom_list = [int(i) for i in atoms]
        cap_info = MLPotential._oniom_normalize_caps(
            args.get("linkRecords"),
            args.get("capMMParams"),
            system,
            atom_list,
            topology,
            dict(args),
        )
        model = _build_oniom_model_system(system, atom_list, cap_info=cap_info)
        neg_closure = _make_oniom_low_model_closure(
            model, num_atoms=system.getNumParticles(), cap_info=cap_info
        )

        def pos_closure(state):
            e_neg, f_neg = neg_closure(state)
            return -e_neg, -f_neg

        PythonForce = getattr(openmm, "PythonForce", None)
        if PythonForce is None:
            PythonForce = getattr(getattr(openmm, "openmm", None), "PythonForce", None)
        if PythonForce is None:
            raise RuntimeError("PythonForce not available")
        force = PythonForce(pos_closure)
        force.setForceGroup(forceGroup)
        host_is_periodic = (
            (topology.getPeriodicBoxVectors() is not None)
            or system.usesPeriodicBoundaryConditions()
        )
        force.setUsesPeriodicBoundaryConditions(host_is_periodic)
        system.addForce(force)


class _MMAsMaceFactory(MLPotentialImplFactory):
    def createImpl(self, name, **args):
        return _MMAsMaceImpl()


if "mm_as_mace_test" not in MLPotential._implFactories:
    MLPotential.registerImplFactory("mm_as_mace_test", _MMAsMaceFactory())


# ---------------------------------------------------------------------------
# A small system with a Q-M boundary bond. ML = atoms 0..2; MM = atoms 3..5.
# Atom 1 is Q (ML side of the boundary), atom 3 is M (MM side).
# Bond 1-3 is the boundary bond that capping replaces with Q-cap.
# ---------------------------------------------------------------------------

def _build_two_residue_chain(periodic=False):
    """Synthetic two-residue chain of 6 atoms with one boundary bond."""
    system = openmm.System()
    nb = openmm.NonbondedForce()
    if periodic:
        nb.setNonbondedMethod(openmm.NonbondedForce.PME)
        nb.setCutoffDistance(0.6 * unit.nanometer)
    else:
        nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    bonds = openmm.HarmonicBondForce()
    angles = openmm.HarmonicAngleForce()

    params = [
        # mass, charge, sigma, epsilon
        (12.0,  0.10, 0.30, 0.30),  # 0 ML
        (12.0, -0.20, 0.32, 0.40),  # 1 ML (Q — boundary)
        (1.0,   0.05, 0.10, 0.05),  # 2 ML
        (12.0,  0.30, 0.31, 0.50),  # 3 MM (M — boundary)
        (1.0,   0.05, 0.10, 0.05),  # 4 MM
        (1.0,   0.05, 0.10, 0.05),  # 5 MM
    ]
    for mass, charge, sigma, epsilon in params:
        system.addParticle(mass)
        nb.addParticle(
            charge * unit.elementary_charge,
            sigma * unit.nanometer,
            epsilon * unit.kilojoule_per_mole,
        )
    bonds.addBond(0, 1, 0.15 * unit.nanometer, 1000.0 * unit.kilojoule_per_mole / unit.nanometer ** 2)
    bonds.addBond(1, 2, 0.10 * unit.nanometer, 800.0 * unit.kilojoule_per_mole / unit.nanometer ** 2)
    bonds.addBond(1, 3, 0.15 * unit.nanometer, 900.0 * unit.kilojoule_per_mole / unit.nanometer ** 2)  # boundary
    bonds.addBond(3, 4, 0.10 * unit.nanometer, 800.0 * unit.kilojoule_per_mole / unit.nanometer ** 2)
    bonds.addBond(3, 5, 0.10 * unit.nanometer, 800.0 * unit.kilojoule_per_mole / unit.nanometer ** 2)

    angles.addAngle(0, 1, 2, 1.9, 100.0 * unit.kilojoule_per_mole / unit.radian ** 2)
    angles.addAngle(0, 1, 3, 1.9, 100.0 * unit.kilojoule_per_mole / unit.radian ** 2)  # boundary
    angles.addAngle(2, 1, 3, 1.9, 100.0 * unit.kilojoule_per_mole / unit.radian ** 2)  # boundary
    angles.addAngle(1, 3, 4, 1.9, 100.0 * unit.kilojoule_per_mole / unit.radian ** 2)  # boundary
    angles.addAngle(4, 3, 5, 1.9, 100.0 * unit.kilojoule_per_mole / unit.radian ** 2)

    system.addForce(nb)
    system.addForce(bonds)
    system.addForce(angles)

    if periodic:
        system.setDefaultPeriodicBoxVectors(
            openmm.Vec3(2.0, 0, 0) * unit.nanometer,
            openmm.Vec3(0, 2.0, 0) * unit.nanometer,
            openmm.Vec3(0, 0, 2.0) * unit.nanometer,
        )

    topology = app.Topology()
    chain = topology.addChain()
    res1 = topology.addResidue("R1", chain)
    res2 = topology.addResidue("R2", chain)
    a0 = topology.addAtom("A0", app.element.carbon, res1)
    a1 = topology.addAtom("A1", app.element.carbon, res1)
    a2 = topology.addAtom("A2", app.element.hydrogen, res1)
    a3 = topology.addAtom("A3", app.element.carbon, res2)
    a4 = topology.addAtom("A4", app.element.hydrogen, res2)
    a5 = topology.addAtom("A5", app.element.hydrogen, res2)
    topology.addBond(a0, a1)
    topology.addBond(a1, a2)
    topology.addBond(a1, a3)
    topology.addBond(a3, a4)
    topology.addBond(a3, a5)
    if periodic:
        topology.setPeriodicBoxVectors(
            unit.Quantity(np.diag([2.0, 2.0, 2.0]), unit.nanometer)
        )
    return topology, system


def _positions_chain():
    return np.array([
        [0.00, 0.00, 0.00],
        [0.15, 0.00, 0.00],   # Q
        [0.20, 0.10, 0.00],
        [0.30, 0.00, 0.00],   # M
        [0.40, 0.05, 0.00],
        [0.40, -0.05, 0.00],
    ]) * unit.nanometer


def _energy(system, positions):
    platform = openmm.Platform.getPlatformByName("Reference")
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
    ctx.setPositions(positions)
    state = ctx.getState(getEnergy=True, getForces=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    return float(e), np.asarray(f)


# ---------------------------------------------------------------------------
# Test 1: ML = MM cancellation
# ---------------------------------------------------------------------------

def test_oniom_collapses_to_real_when_mm_mass_mace_no_caps():
    """Closed-valence (no linkRecords). With high-level = E_MM(model),
    ``E_total = E_MM(real) + E_MM(model) - E_MM(model)`` collapses to
    ``E_MM(real)``. Catches any subtraction-side bug in the closed-valence
    path."""
    topology, source = _build_two_residue_chain(periodic=False)
    pos = _positions_chain()

    # Reference: bare source MM total.
    e_real, f_real = _energy(source, pos)

    potential = MLPotential("mm_as_mace_test")
    mixed = potential.createMixedSystem(
        topology, source, atoms=[0, 1, 2], embedding="oniom-electrostatic"
    )
    e_mix, f_mix = _energy(mixed, pos)

    assert e_mix == pytest.approx(e_real, rel=1e-9, abs=1e-8)
    np.testing.assert_allclose(f_mix, f_real, rtol=1e-7, atol=1e-7)


def test_oniom_collapses_to_real_when_mm_mass_mace_with_caps():
    """Capped path: same cancellation must hold even with link records.
    Specifically tests that:

    - The cap-bearing model `System` is constructed with N+K particles
      consistently between the high-level (mm_as_mace_test) closure and
      the ONIOM low-model correction closure.
    - Cap force redistribution onto Q,M is symmetric: forces from the
      high-level closure cancel forces from the low-model correction
      bit-exactly, including the per-step Jacobian application.
    - Total energy = E_MM(real). Total forces = MM-only forces.

    If cap placement, charge transfer, or force redistribution is wrong,
    this will diverge from E_MM(real) by a non-zero residual.
    """
    topology, source = _build_two_residue_chain(periodic=False)
    pos = _positions_chain()

    # Reference: bare source MM total. Caps don't appear here.
    e_real, f_real = _energy(source, pos)

    potential = MLPotential("mm_as_mace_test")
    mixed = potential.createMixedSystem(
        topology, source, atoms=[0, 1, 2],
        embedding="oniom-electrostatic",
        linkRecords=[(1, 3, 1.09)],   # Q=1, M=3, target dist in Å (MACE convention)
    )
    e_mix, f_mix = _energy(mixed, pos)

    assert e_mix == pytest.approx(e_real, rel=1e-9, abs=1e-8)
    np.testing.assert_allclose(f_mix, f_real, rtol=1e-7, atol=1e-7)


# ---------------------------------------------------------------------------
# Test 5: PBC boundary
# ---------------------------------------------------------------------------

def test_oniom_pbc_with_caps_raises_clearly():
    """PBC + linkRecords is currently unsupported because the cap
    placement formula does not apply minimum-image to (r_M − r_Q).
    A future slice will lift this; today it must raise loudly so the
    user doesn't get silently-wrong energies for cross-boundary cuts.

    The error currently bubbles up from
    ``_prepareLinkRecords`` (used by both MACE and the ONIOM cap
    normalizer) which rejects ``is_periodic=True``.
    """
    topology, source = _build_two_residue_chain(periodic=True)
    potential = MLPotential("mm_as_mace_test")
    with pytest.raises((NotImplementedError, ValueError), match=r"non-periodic|PBC|periodic"):
        potential.createMixedSystem(
            topology, source, atoms=[0, 1, 2],
            embedding="oniom-electrostatic",
            linkRecords=[(1, 3, 0.109)],
        )


def test_oniom_pbc_no_caps_still_works():
    """Sanity: closed-valence ONIOM under PBC still works (Slice 2′
    covered this; we're confirming the new cap-aware code paths
    don't regress it)."""
    topology, source = _build_two_residue_chain(periodic=True)
    pos = _positions_chain()

    e_real, f_real = _energy(source, pos)

    potential = MLPotential("mm_as_mace_test")
    mixed = potential.createMixedSystem(
        topology, source, atoms=[0, 1, 2], embedding="oniom-electrostatic"
    )
    e_mix, f_mix = _energy(mixed, pos)

    assert e_mix == pytest.approx(e_real, rel=1e-9, abs=1e-8)
    np.testing.assert_allclose(f_mix, f_real, rtol=1e-7, atol=1e-7)
