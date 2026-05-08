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


def test_create_mixed_system_with_oniom_raises_not_implemented():
    """The mode is recognized at the API boundary and raises a clear
    NotImplementedError pointing to the plan doc."""
    topology, system = _minimal_system_and_topology()
    potential = MLPotential("noop_oniom_test")
    with pytest.raises(NotImplementedError, match="oniom-electrostatic"):
        potential.createMixedSystem(
            topology, system, [0, 1], embedding="oniom-electrostatic"
        )


def test_oniom_with_interpolate_true_raises_value_error():
    """interpolate=True must raise ValueError BEFORE the
    NotImplementedError so a future implementation cannot silently
    fall through into the existing CustomCVForce interpolation path
    (Codex finding #8 in the redesign plan).
    """
    topology, system = _minimal_system_and_topology()
    potential = MLPotential("noop_oniom_test")
    with pytest.raises(ValueError, match="interpolate=True"):
        potential.createMixedSystem(
            topology,
            system,
            [0, 1],
            embedding="oniom-electrostatic",
            interpolate=True,
        )


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
