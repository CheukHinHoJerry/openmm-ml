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
    """Slice 2′: closed-valence ONIOM returns a working System."""
    topology, system = _minimal_system_and_topology()
    potential = MLPotential("noop_oniom_test")
    new_system = potential.createMixedSystem(
        topology, system, [0, 1], embedding="oniom-electrostatic"
    )
    assert isinstance(new_system, openmm.System)
    assert new_system.getNumParticles() == system.getNumParticles()


def test_create_mixed_system_with_oniom_link_records_still_raises():
    """Slice 2′: capped ONIOM (linkRecords != None) is still gated to
    Slice 3′."""
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


# ---------------------------------------------------------------------------
# Reviewer-flagged correctness fixes for the closed-valence Slice 2′ path.
# ---------------------------------------------------------------------------

def _system_with_two_nonbonded_forces():
    """Source System with TWO ``NonbondedForce`` instances (e.g.,
    alchemical / layered setup). Both must contribute to E_MM(model)."""
    system = openmm.System()
    nb1 = openmm.NonbondedForce()
    nb2 = openmm.NonbondedForce()
    nb1.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    nb2.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for charge_e in (0.5, -0.3, 0.1):
        system.addParticle(12.0)
        nb1.addParticle(
            charge_e * unit.elementary_charge,
            0.3 * unit.nanometer,
            0.2 * unit.kilojoule_per_mole,
        )
        nb2.addParticle(
            (0.5 * charge_e) * unit.elementary_charge,
            0.32 * unit.nanometer,
            0.15 * unit.kilojoule_per_mole,
        )
    system.addForce(nb1)
    system.addForce(nb2)
    topology = app.Topology()
    chain = topology.addChain()
    res = topology.addResidue("MOL", chain)
    for i in range(3):
        topology.addAtom(f"A{i}", app.element.carbon, res)
    return topology, system


def test_oniom_handles_multiple_nonbonded_forces():
    """The model `System` must include EVERY ``NonbondedForce`` from the
    source, not just the first. Layered/alchemical setups commonly have
    more than one. Reviewer P1 finding."""
    topology, system = _system_with_two_nonbonded_forces()
    potential = MLPotential("noop_oniom_test")
    new_system = potential.createMixedSystem(
        topology, system, [0, 1], embedding="oniom-electrostatic"
    )
    assert isinstance(new_system, openmm.System)
    # Smoke: a Context can be instantiated and energy queried without
    # raising. Architectural validation only — the parity test in
    # TestOniomParity.py covers numerical agreement on the real PME
    # workflow.
    platform = openmm.Platform.getPlatformByName("Reference")
    ctx = openmm.Context(new_system, openmm.VerletIntegrator(0.001), platform)
    import numpy as np
    ctx.setPositions(
        np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]])
        * unit.nanometer
    )
    state = ctx.getState(getEnergy=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert isinstance(e, float)


def _system_with_custom_nonbonded():
    """Source System with a ``CustomNonbondedForce`` *and* a
    ``NonbondedForce``. The model `System` must mirror the ML-internal
    contribution of both."""
    system = openmm.System()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    cnb = openmm.CustomNonbondedForce("0.5*r")  # synthetic functional form
    cnb.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
    for _ in range(3):
        system.addParticle(12.0)
        nb.addParticle(
            0.1 * unit.elementary_charge,
            0.3 * unit.nanometer,
            0.2 * unit.kilojoule_per_mole,
        )
        cnb.addParticle([])
    system.addForce(nb)
    system.addForce(cnb)
    topology = app.Topology()
    chain = topology.addChain()
    res = topology.addResidue("MOL", chain)
    for i in range(3):
        topology.addAtom(f"A{i}", app.element.carbon, res)
    return topology, system


def test_oniom_handles_custom_nonbonded_force():
    """``CustomNonbondedForce`` in the source must show up in
    E_MM(model) (with ML-ML excluded, mirroring the existing
    electrostatic-mode behavior). Reviewer P1 finding — without this,
    ML-internal CustomNonbondedForce energy stays in the host but is
    never subtracted, silently double-counting.

    The CustomCVForce that holds the opposite-sign clones lives in the
    model `System` (inside the PythonForce closure), so we verify it
    by calling the underlying builder directly."""
    from openmmml.mlpotential import _build_oniom_model_system

    _, system = _system_with_custom_nonbonded()
    model = _build_oniom_model_system(system, [0, 1])

    cvs = [f for f in model.getForces() if isinstance(f, openmm.CustomCVForce)]
    assert len(cvs) == 1
    cv = cvs[0]
    cv_names = [
        cv.getCollectiveVariableName(i) for i in range(cv.getNumCollectiveVariables())
    ]
    # Two source nonbonded forces (1 NB + 1 CNB) → 2 (a, b) pairs → 4 CVs.
    assert len(cv_names) == 4
    energy = cv.getEnergyFunction()
    # Energy must subtract both pairs.
    assert "nb_a_0 - nb_b_0" in energy
    assert "nb_a_1 - nb_b_1" in energy


def _system_with_unsupported_nonbonded():
    """Source System with a CustomGBForce — explicitly out of scope."""
    system = openmm.System()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for _ in range(3):
        system.addParticle(12.0)
        nb.addParticle(
            0.0 * unit.elementary_charge,
            0.3 * unit.nanometer,
            0.2 * unit.kilojoule_per_mole,
        )
    system.addForce(nb)
    cgb_cls = getattr(openmm, "CustomGBForce", None)
    if cgb_cls is None:
        return None  # OpenMM lacks CustomGBForce in this build; skip.
    cgb = cgb_cls()
    for _ in range(3):
        cgb.addParticle([])
    system.addForce(cgb)
    topology = app.Topology()
    chain = topology.addChain()
    res = topology.addResidue("MOL", chain)
    for i in range(3):
        topology.addAtom(f"A{i}", app.element.carbon, res)
    return topology, system


def test_oniom_custom_nonbonded_ml_internal_cancels_numerically():
    """End-to-end numerical check: with a CustomNonbondedForce in the
    source, the model `System`'s ``nb_a - nb_b`` for that force MUST
    equal exactly the ML-internal energy of the source CustomNonbondedForce
    (i.e., what the existing electrostatic mode would remove via
    exclusions). Without the fix, this would be 0 and the cancellation
    would silently fail."""
    import numpy as np
    from openmmml.mlpotential import _build_oniom_model_system

    _, system = _system_with_custom_nonbonded()
    model = _build_oniom_model_system(system, [0, 1])
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]]
    ) * unit.nanometer

    platform = openmm.Platform.getPlatformByName("Reference")

    # Energy of the model System (what the closure would return as -E).
    ctx_model = openmm.Context(model, openmm.VerletIntegrator(0.001), platform)
    ctx_model.setPositions(positions)
    e_model = (
        ctx_model.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )

    # Reference: hand-build a System containing only the source's
    # CustomNonbondedForce with ML-ML restricted (interaction group
    # only on ML atoms). That equals the ML-internal contribution.
    src_cnb = next(
        f for f in system.getForces() if isinstance(f, openmm.CustomNonbondedForce)
    )
    src_nb = next(
        f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)
    )
    cnb_xml = openmm.XmlSerializer.serialize(src_cnb)
    nb_xml = openmm.XmlSerializer.serialize(src_nb)

    ref_system = openmm.System()
    for i in range(system.getNumParticles()):
        ref_system.addParticle(system.getParticleMass(i))

    # Reference contribution from the regular NonbondedForce, ML-internal
    # only: same opposite-sign trick on the regular NB.
    nb_ref_a = openmm.XmlSerializer.deserialize(nb_xml)
    nb_ref_b = openmm.XmlSerializer.deserialize(nb_xml)
    from openmmml.mlpotential import _oniom_apply_nonbonded_surgery
    _oniom_apply_nonbonded_surgery(nb_ref_b, {0, 1})
    cnb_ref_a = openmm.XmlSerializer.deserialize(cnb_xml)
    cnb_ref_b = openmm.XmlSerializer.deserialize(cnb_xml)
    from openmmml.mlpotential import _oniom_apply_custom_nonbonded_surgery
    _oniom_apply_custom_nonbonded_surgery(cnb_ref_b, {0, 1})
    cv = openmm.CustomCVForce(
        "(nb_a - nb_b) + (cnb_a - cnb_b)"
    )
    cv.addCollectiveVariable("nb_a", nb_ref_a)
    cv.addCollectiveVariable("nb_b", nb_ref_b)
    cv.addCollectiveVariable("cnb_a", cnb_ref_a)
    cv.addCollectiveVariable("cnb_b", cnb_ref_b)
    ref_system.addForce(cv)

    ctx_ref = openmm.Context(ref_system, openmm.VerletIntegrator(0.001), platform)
    ctx_ref.setPositions(positions)
    e_ref = (
        ctx_ref.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )

    # Same expression, same particles, same positions → bit-exact agreement.
    assert e_model == pytest.approx(e_ref, rel=1e-12, abs=1e-12)
    # The CustomNonbondedForce contribution must be non-zero (otherwise
    # the test isn't actually exercising the new code path).
    assert abs(e_model) > 0


def test_oniom_rejects_unsupported_nonbonded_like_force():
    """``CustomGBForce`` (and other nonbonded-like classes the builder
    doesn't handle) must raise ``NotImplementedError`` rather than be
    silently dropped from the model `System`."""
    payload = _system_with_unsupported_nonbonded()
    if payload is None:
        pytest.skip("openmm.CustomGBForce not available")
    topology, system = payload
    potential = MLPotential("noop_oniom_test")
    with pytest.raises(NotImplementedError, match="CustomGBForce"):
        potential.createMixedSystem(
            topology, system, [0, 1], embedding="oniom-electrostatic"
        )


# ---------------------------------------------------------------------------
# Parity vs. the existing additive `electrostatic` mode for the P1 paths.
#
# Both modes implement the same ONIOM-EE bookkeeping; the existing mode
# bakes the subtraction into a host-System mutation, the new mode adds
# an additive correction PythonForce. With a no-op MLPotentialImpl
# (zero MACE contribution) the two should produce IDENTICAL total
# energies and forces.
# ---------------------------------------------------------------------------

def _energy_and_forces(system, positions):
    import numpy as np
    platform = openmm.Platform.getPlatformByName("Reference")
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
    ctx.setPositions(positions)
    state = ctx.getState(getEnergy=True, getForces=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    return float(e), np.asarray(f)


def _build_two_nonbonded_topology_system_for_parity():
    """3-atom system with two NonbondedForce instances. Matches the shape
    of an alchemical / layered MM stack."""
    system = openmm.System()
    nb1 = openmm.NonbondedForce()
    nb2 = openmm.NonbondedForce()
    nb1.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    nb2.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for charge_e in (0.5, -0.3, 0.1):
        system.addParticle(12.0)
        nb1.addParticle(
            charge_e * unit.elementary_charge,
            0.3 * unit.nanometer,
            0.2 * unit.kilojoule_per_mole,
        )
        nb2.addParticle(
            (0.5 * charge_e) * unit.elementary_charge,
            0.32 * unit.nanometer,
            0.15 * unit.kilojoule_per_mole,
        )
    system.addForce(nb1)
    system.addForce(nb2)
    topology = app.Topology()
    chain = topology.addChain()
    res = topology.addResidue("MOL", chain)
    for i in range(3):
        topology.addAtom(f"A{i}", app.element.carbon, res)
    return topology, system


def _build_nb_plus_custom_nb_topology_system_for_parity():
    """3-atom system with a NonbondedForce and a CustomNonbondedForce."""
    system = openmm.System()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    cnb = openmm.CustomNonbondedForce("0.5*r")  # synthetic
    cnb.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
    for charge_e in (0.4, -0.2, 0.05):
        system.addParticle(12.0)
        nb.addParticle(
            charge_e * unit.elementary_charge,
            0.3 * unit.nanometer,
            0.2 * unit.kilojoule_per_mole,
        )
        cnb.addParticle([])
    system.addForce(nb)
    system.addForce(cnb)
    topology = app.Topology()
    chain = topology.addChain()
    res = topology.addResidue("MOL", chain)
    for i in range(3):
        topology.addAtom(f"A{i}", app.element.carbon, res)
    return topology, system


def _positions_3atoms():
    import numpy as np
    return (
        np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]])
        * unit.nanometer
    )


def test_oniom_matches_electrostatic_mode_two_nonbonded_forces():
    """Numerical parity: with a no-op MACE and two source NonbondedForces,
    the new ONIOM mode and the existing additive `electrostatic` mode
    must produce identical totals. Pins the multiple-NonbondedForce
    P1 fix against the additive-mode reference."""
    topology, system_a = _build_two_nonbonded_topology_system_for_parity()
    topology, system_b = _build_two_nonbonded_topology_system_for_parity()
    potential = MLPotential("noop_oniom_test")
    pos = _positions_3atoms()

    elec = potential.createMixedSystem(
        topology, system_a, [0, 1], embedding="electrostatic"
    )
    e_elec, f_elec = _energy_and_forces(elec, pos)

    oniom = potential.createMixedSystem(
        topology, system_b, [0, 1], embedding="oniom-electrostatic"
    )
    e_oniom, f_oniom = _energy_and_forces(oniom, pos)

    assert e_oniom == pytest.approx(e_elec, rel=1e-9, abs=1e-9)
    import numpy as np
    np.testing.assert_allclose(f_oniom, f_elec, rtol=1e-9, atol=1e-9)


def test_oniom_matches_electrostatic_mode_with_custom_nonbonded():
    """Numerical parity: with a no-op MACE and a source containing both
    NonbondedForce and CustomNonbondedForce, the new ONIOM mode must
    match the existing additive `electrostatic` mode (which removes
    ML-internal CustomNonbondedForce energy via ML-ML exclusions on
    the host)."""
    topology, system_a = _build_nb_plus_custom_nb_topology_system_for_parity()
    topology, system_b = _build_nb_plus_custom_nb_topology_system_for_parity()
    potential = MLPotential("noop_oniom_test")
    pos = _positions_3atoms()

    elec = potential.createMixedSystem(
        topology, system_a, [0, 1], embedding="electrostatic"
    )
    e_elec, f_elec = _energy_and_forces(elec, pos)

    oniom = potential.createMixedSystem(
        topology, system_b, [0, 1], embedding="oniom-electrostatic"
    )
    e_oniom, f_oniom = _energy_and_forces(oniom, pos)

    assert e_oniom == pytest.approx(e_elec, rel=1e-9, abs=1e-9)
    import numpy as np
    np.testing.assert_allclose(f_oniom, f_elec, rtol=1e-9, atol=1e-9)


def test_oniom_matches_electrostatic_mode_minimal_no_op():
    """Sanity baseline: even on the trivial 3-atom system used by the
    other API tests, with no MACE contribution, both modes give the
    same MM-only total. Catches anything global about the construction
    paths drifting apart."""
    topology, system_a = _minimal_system_and_topology()
    _, system_b = _minimal_system_and_topology()
    potential = MLPotential("noop_oniom_test")
    pos = _positions_3atoms()

    elec = potential.createMixedSystem(
        topology, system_a, [0, 1], embedding="electrostatic"
    )
    e_elec, f_elec = _energy_and_forces(elec, pos)

    oniom = potential.createMixedSystem(
        topology, system_b, [0, 1], embedding="oniom-electrostatic"
    )
    e_oniom, f_oniom = _energy_and_forces(oniom, pos)

    assert e_oniom == pytest.approx(e_elec, rel=1e-9, abs=1e-9)
    import numpy as np
    np.testing.assert_allclose(f_oniom, f_elec, rtol=1e-9, atol=1e-9)
