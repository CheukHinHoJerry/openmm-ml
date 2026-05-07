"""Tests for embedding='oniom-electrostatic' (Slice 1: API skeleton).

Slice 1 of docs/codex-plans/electrostatic-oniom-implementation-plan.md.

The mode is reserved at the API boundary but not yet implemented; these
tests pin that surface so subsequent slices can extend behavior without
breaking existing callers.
"""
from __future__ import annotations

import openmm
import openmm.app as app
import openmm.unit as unit
import pytest

from openmmml import MLPotential
from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory
from openmmml.models.macepotential import _should_use_mm_embedding


# ---------------------------------------------------------------------------
# No-op MLPotentialImpl for testing the pre-MACE API surface
# ---------------------------------------------------------------------------

class _NoopImpl(MLPotentialImpl):
    def addForces(self, topology, system, atoms, forceGroup, **args):
        return


class _NoopFactory(MLPotentialImplFactory):
    def createImpl(self, name, **args):
        return _NoopImpl()


MLPotential.registerImplFactory("noop_oniom_test", _NoopFactory())


def _minimal_system_and_topology():
    system = openmm.System()
    nonbonded = openmm.NonbondedForce()
    for _ in range(3):
        system.addParticle(12.0)
        nonbonded.addParticle(
            0.0 * unit.elementary_charge,
            0.3 * unit.nanometer,
            0.2 * unit.kilojoule_per_mole,
        )
    system.addForce(nonbonded)
    topology = app.Topology()
    chain = topology.addChain()
    res = topology.addResidue("MOL", chain)
    for i in range(3):
        topology.addAtom(f"A{i}", app.element.carbon, res)
    return topology, system


def test_create_mixed_system_with_oniom_closed_valence_returns_system():
    """Closed-valence ONIOM (linkRecords=None) is implemented as of Slice 3
    and returns an `openmm.System`."""
    topology, system = _minimal_system_and_topology()
    potential = MLPotential("noop_oniom_test")
    new_system = potential.createMixedSystem(
        topology, system, [0, 1], embedding="oniom-electrostatic"
    )
    assert isinstance(new_system, openmm.System)
    assert new_system.getNumParticles() == system.getNumParticles()


def test_create_mixed_system_with_oniom_link_records_raises_not_implemented():
    """Capped (linkRecords != None) ONIOM is reserved for Slice 4."""
    topology, system = _minimal_system_and_topology()
    potential = MLPotential("noop_oniom_test")
    with pytest.raises(NotImplementedError, match="link-atom"):
        potential.createMixedSystem(
            topology,
            system,
            [0, 1],
            embedding="oniom-electrostatic",
            linkRecords=[(0, 2, 0.109)],
        )


def test_create_mixed_system_with_oniom_no_atoms_raises_value_error():
    """Caller must specify ml_atoms."""
    topology, system = _minimal_system_and_topology()
    potential = MLPotential("noop_oniom_test")
    with pytest.raises(ValueError, match="ml-atoms"):
        potential.createMixedSystem(
            topology, system, None, embedding="oniom-electrostatic"
        )


def test_create_mixed_system_with_oniom_interpolate_raises_value_error():
    """ONIOM does not support `interpolate=True`."""
    topology, system = _minimal_system_and_topology()
    potential = MLPotential("noop_oniom_test")
    with pytest.raises(ValueError, match="interpolate=True"):
        potential.createMixedSystem(
            topology, system, [0, 1], embedding="oniom-electrostatic", interpolate=True
        )


def test_oniom_zeros_ml_charges_and_keeps_lj():
    """The Slice 3 host surgery zeros ML particle charges and ML-* exception
    chargeProd. Sigma/epsilon are preserved on every particle and exception."""
    topology, system = _minimal_system_and_topology()
    # Add a non-zero charge so we can prove zeroing happens.
    nb = system.getForce(0)
    nb.setParticleParameters(
        0, 0.5 * unit.elementary_charge, 0.30 * unit.nanometer, 0.20 * unit.kilojoule_per_mole
    )
    nb.setParticleParameters(
        1, -0.3 * unit.elementary_charge, 0.32 * unit.nanometer, 0.25 * unit.kilojoule_per_mole
    )
    nb.setParticleParameters(
        2, 0.1 * unit.elementary_charge, 0.31 * unit.nanometer, 0.65 * unit.kilojoule_per_mole
    )
    potential = MLPotential("noop_oniom_test")
    new_system = potential.createMixedSystem(
        topology, system, [0, 1], embedding="oniom-electrostatic"
    )
    new_nb = next(
        f for f in new_system.getForces() if isinstance(f, openmm.NonbondedForce)
    )
    # ML particles zeroed.
    for i in (0, 1):
        q, sigma, eps = new_nb.getParticleParameters(i)
        assert q.value_in_unit(unit.elementary_charge) == pytest.approx(0.0, abs=1e-12)
        assert sigma.value_in_unit(unit.nanometer) > 0
    # MM particle untouched.
    q2, _, _ = new_nb.getParticleParameters(2)
    assert q2.value_in_unit(unit.elementary_charge) == pytest.approx(0.1, abs=1e-12)
    # ML-* exceptions exist with chargeProd=0.
    excs = {}
    for i in range(new_nb.getNumExceptions()):
        p1, p2, cp, sigma, eps = new_nb.getExceptionParameters(i)
        excs[tuple(sorted((int(p1), int(p2))))] = (
            cp.value_in_unit(unit.elementary_charge ** 2),
            sigma.value_in_unit(unit.nanometer),
            eps.value_in_unit(unit.kilojoule_per_mole),
        )
    assert excs[(0, 1)][0] == pytest.approx(0.0, abs=1e-15)
    assert excs[(0, 2)][0] == pytest.approx(0.0, abs=1e-15)
    assert excs[(1, 2)][0] == pytest.approx(0.0, abs=1e-15)
    # ML-ML LJ retained: non-zero epsilon (low-model will subtract it).
    assert excs[(0, 1)][2] > 0


def test_oniom_low_model_correction_force_added():
    """The low-model correction is added as a `CustomCVForce` with the
    expected name/structure."""
    topology, system = _minimal_system_and_topology()
    potential = MLPotential("noop_oniom_test")
    new_system = potential.createMixedSystem(
        topology, system, [0, 1], embedding="oniom-electrostatic"
    )
    cv_forces = [f for f in new_system.getForces() if isinstance(f, openmm.CustomCVForce)]
    assert len(cv_forces) == 1
    cv = cv_forces[0]
    assert cv.getEnergyFunction().startswith("-1*")


def test_unknown_embedding_still_rejected_at_mace_layer():
    """Unrelated bad values must still raise ValueError, not slip through
    as ONIOM."""

    class _Polar:
        __class__ = type("PolarMACE", (), {})  # placeholder used by name check

    class _PolarMACE:
        pass

    _PolarMACE.__name__ = "PolarMACE"
    instance = _PolarMACE()

    with pytest.raises(ValueError, match="Unsupported embedding mode"):
        _should_use_mm_embedding(instance, [0, 1], "qmmm-cool-mode")


def test_should_use_mm_embedding_recognizes_oniom():
    """ONIOM is in the same MM-embedding-modes bucket as electrostatic."""

    class _PolarMACE:
        pass

    _PolarMACE.__name__ = "PolarMACE"
    instance = _PolarMACE()

    assert _should_use_mm_embedding(instance, [0, 1], "oniom-electrostatic")
    assert _should_use_mm_embedding(instance, [0, 1], "electrostatic")
    assert not _should_use_mm_embedding(instance, None, "oniom-electrostatic")
    assert not _should_use_mm_embedding(instance, [0, 1], "mechanical")


def test_existing_electrostatic_mode_still_runs():
    """No regression: the existing electrostatic mode path continues to work
    on the same minimal system."""
    topology, system = _minimal_system_and_topology()
    potential = MLPotential("noop_oniom_test")
    new_system = potential.createMixedSystem(
        topology, system, [0, 1], embedding="electrostatic"
    )
    assert isinstance(new_system, openmm.System)
    assert new_system.getNumParticles() == 3


def test_existing_mechanical_mode_still_runs():
    """No regression: the default mechanical-embedding path is unchanged."""
    topology, system = _minimal_system_and_topology()
    potential = MLPotential("noop_oniom_test")
    new_system = potential.createMixedSystem(topology, system, [0, 1])
    assert isinstance(new_system, openmm.System)
    assert new_system.getNumParticles() == 3
