"""Slice 2 unit tests for ``OniomLowModelBuilder``.

The builder produces the MM-level energy of the ML region as a set of
ordinary OpenMM `Force` objects (the closed-valence / no-link-atom
case). These tests exercise the static path:

- Bonded copies pick up only ML-internal entries.
- ML-internal LJ + Coulomb match the source NonbondedForce (NoCutoff).
- ML-MM Coulomb background matches the corresponding term in the source
  NonbondedForce (NoCutoff).
- The builder leaves the source ``System`` untouched.

PBC + Ewald-split parity with PME is deferred to Slice 3 (which can
work around it by zeroing ML charges in the host system) and Slice 5
(which makes the stack purely additive). Slice 2 is verified with
NoCutoff so the comparison is exact.
"""
from __future__ import annotations

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
import pytest

from openmmml.embedding import OniomLowModelBuilder


# ---------------------------------------------------------------------------
# Test system construction (non-periodic so 1/r matches exactly).
# ---------------------------------------------------------------------------

# 5 atoms: ML = {0, 1, 2}, MM = {3, 4}
_PARAMS = [
    # mass, charge_e, sigma_nm, epsilon_kj
    (12.0, 0.6, 0.30, 0.20),   # ML 0
    (12.0, -0.4, 0.32, 0.25),  # ML 1
    (1.0, 0.1, 0.25, 0.10),    # ML 2
    (16.0, -0.8, 0.31, 0.65),  # MM 3
    (1.0, 0.4, 0.10, 0.05),    # MM 4
]
_ML_ATOMS = [0, 1, 2]
_MM_ATOMS = [3, 4]


def _build_source_system(with_bonded=True, with_ml_mm_exception=False):
    system = openmm.System()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for mass, charge, sigma, epsilon in _PARAMS:
        system.addParticle(mass)
        nb.addParticle(
            charge * unit.elementary_charge,
            sigma * unit.nanometer,
            epsilon * unit.kilojoule_per_mole,
        )
    if with_ml_mm_exception:
        # 1-3 ML-MM exception with explicit (nonzero) sigma/epsilon and
        # nonzero chargeProd — pin that builder honors the exception.
        nb.addException(
            2, 3,
            0.123 * unit.elementary_charge * unit.elementary_charge,
            0.345 * unit.nanometer,
            0.678 * unit.kilojoule_per_mole,
        )
    system.addForce(nb)

    if with_bonded:
        bonds = openmm.HarmonicBondForce()
        bonds.addBond(0, 1, 0.15 * unit.nanometer, 1000.0 * unit.kilojoule_per_mole / unit.nanometer ** 2)  # ML-ML
        bonds.addBond(1, 2, 0.15 * unit.nanometer, 800.0 * unit.kilojoule_per_mole / unit.nanometer ** 2)   # ML-ML
        bonds.addBond(2, 3, 0.15 * unit.nanometer, 600.0 * unit.kilojoule_per_mole / unit.nanometer ** 2)   # ML-MM (boundary)
        bonds.addBond(3, 4, 0.10 * unit.nanometer, 500.0 * unit.kilojoule_per_mole / unit.nanometer ** 2)   # MM-MM
        system.addForce(bonds)

        angles = openmm.HarmonicAngleForce()
        angles.addAngle(0, 1, 2, 1.9, 100.0 * unit.kilojoule_per_mole / unit.radian ** 2)  # ML-ML-ML (internal)
        angles.addAngle(1, 2, 3, 1.9, 100.0 * unit.kilojoule_per_mole / unit.radian ** 2)  # ML-ML-MM (boundary)
        angles.addAngle(2, 3, 4, 1.9, 100.0 * unit.kilojoule_per_mole / unit.radian ** 2)  # ML-MM-MM (boundary)
        system.addForce(angles)

        torsions = openmm.PeriodicTorsionForce()
        torsions.addTorsion(0, 1, 2, 3, 2, 0.0, 5.0 * unit.kilojoule_per_mole)  # ML-ML-ML-MM (boundary)
        system.addForce(torsions)
    return system


def _positions():
    return np.array([
        [0.0, 0.0, 0.0],
        [0.15, 0.0, 0.0],
        [0.30, 0.05, 0.0],
        [0.50, 0.05, 0.0],
        [0.60, 0.05, 0.0],
    ]) * unit.nanometer


def _energy_of(system, positions):
    platform = openmm.Platform.getPlatformByName("Reference")
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
    ctx.setPositions(positions)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )


def _strip_to_force_types(system, allowed):
    for idx in reversed(range(system.getNumForces())):
        if not isinstance(system.getForce(idx), allowed):
            system.removeForce(idx)
    return system


# ---------------------------------------------------------------------------
# Bonded tests
# ---------------------------------------------------------------------------

def test_internal_bonded_keeps_only_ml_internal():
    system = _build_source_system()
    builder = OniomLowModelBuilder(system, _ML_ATOMS)
    forces = builder.internal_bonded_forces()
    by_type = {type(f): f for f in forces}

    bonds = by_type[openmm.HarmonicBondForce]
    bond_pairs = {tuple(sorted((int(bonds.getBondParameters(i)[0]), int(bonds.getBondParameters(i)[1]))))
                  for i in range(bonds.getNumBonds())}
    assert bond_pairs == {(0, 1), (1, 2)}

    angles = by_type[openmm.HarmonicAngleForce]
    angle_triples = {(int(angles.getAngleParameters(i)[0]),
                       int(angles.getAngleParameters(i)[1]),
                       int(angles.getAngleParameters(i)[2]))
                      for i in range(angles.getNumAngles())}
    assert angle_triples == {(0, 1, 2)}

    # PeriodicTorsionForce had only a boundary torsion (0-1-2-3); ML-internal
    # subset is empty, so the builder drops the empty force entirely.
    assert openmm.PeriodicTorsionForce not in by_type


def test_internal_bonded_energy_matches_source_minus_complement():
    """Energy of the internal-bonded forces equals (source bonded energy) -
    (source bonded energy with all ML-internal bonded entries dropped)."""
    source = _build_source_system()
    positions = _positions()

    # Energy of the source's bonded forces alone.
    only_bonded = _build_source_system()
    _strip_to_force_types(
        only_bonded,
        (openmm.HarmonicBondForce, openmm.HarmonicAngleForce, openmm.PeriodicTorsionForce),
    )
    e_source_bonded = _energy_of(only_bonded, positions)

    # Energy of the internal-only bonded forces alone.
    internal = openmm.System()
    for _ in range(source.getNumParticles()):
        internal.addParticle(1.0)
    builder = OniomLowModelBuilder(source, _ML_ATOMS)
    for f in builder.internal_bonded_forces():
        internal.addForce(f)
    e_internal = _energy_of(internal, positions)

    # Energy of the boundary-only bonded (everything not all-ML).
    complement = openmm.System()
    for _ in range(source.getNumParticles()):
        complement.addParticle(1.0)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(2, 3, 0.15 * unit.nanometer, 600.0 * unit.kilojoule_per_mole / unit.nanometer ** 2)
    bonds.addBond(3, 4, 0.10 * unit.nanometer, 500.0 * unit.kilojoule_per_mole / unit.nanometer ** 2)
    complement.addForce(bonds)
    angles = openmm.HarmonicAngleForce()
    angles.addAngle(1, 2, 3, 1.9, 100.0 * unit.kilojoule_per_mole / unit.radian ** 2)
    angles.addAngle(2, 3, 4, 1.9, 100.0 * unit.kilojoule_per_mole / unit.radian ** 2)
    complement.addForce(angles)
    torsions = openmm.PeriodicTorsionForce()
    torsions.addTorsion(0, 1, 2, 3, 2, 0.0, 5.0 * unit.kilojoule_per_mole)
    complement.addForce(torsions)
    e_complement = _energy_of(complement, positions)

    assert e_internal + e_complement == pytest.approx(e_source_bonded, rel=1e-9, abs=1e-9)


# ---------------------------------------------------------------------------
# ML-internal LJ + Coulomb tests
# ---------------------------------------------------------------------------

def test_internal_nonbonded_matches_source_ml_only():
    """Energy of the builder's ML-internal NonbondedForce CustomBondForce
    equals the source NonbondedForce energy on a system where MM atom charges
    and LJ are zeroed (so only ML-internal pair sums survive)."""
    source = _build_source_system()
    positions = _positions()

    # ML-internal-only reference: zero MM particles' charge/sigma/epsilon and
    # add explicit ML-MM exclusions so only ML-ML pairs contribute.
    ref = _build_source_system()
    _strip_to_force_types(ref, openmm.NonbondedForce)
    nb_ref = ref.getForce(0)
    for i in _MM_ATOMS:
        _, sigma, epsilon = nb_ref.getParticleParameters(i)
        nb_ref.setParticleParameters(
            i, 0.0 * unit.elementary_charge, sigma, 0.0 * unit.kilojoule_per_mole
        )
    # Zero ML-MM cross interactions via exception.
    for ml in _ML_ATOMS:
        for mm in _MM_ATOMS:
            nb_ref.addException(
                ml, mm,
                0.0 * unit.elementary_charge * unit.elementary_charge,
                0.1 * unit.nanometer,
                0.0 * unit.kilojoule_per_mole,
                replace=True,
            )
    e_ref = _energy_of(ref, positions)

    builder = OniomLowModelBuilder(source, _ML_ATOMS)
    cbf = builder.internal_nonbonded_force()
    assert cbf is not None

    test_system = openmm.System()
    for _ in range(source.getNumParticles()):
        test_system.addParticle(1.0)
    test_system.addForce(cbf)
    e_test = _energy_of(test_system, positions)

    assert e_test == pytest.approx(e_ref, rel=1e-9, abs=1e-9)


def test_internal_nonbonded_skips_pure_ml_zero_charge_zero_eps_pairs():
    """If an ML-ML pair has both chargeProd=0 and epsilon=0 (explicit exception),
    it must not be added to the CustomBondForce."""
    system = _build_source_system()
    nb = system.getForce(0)
    nb.addException(
        0, 1,
        0.0 * unit.elementary_charge * unit.elementary_charge,
        0.3 * unit.nanometer,
        0.0 * unit.kilojoule_per_mole,
    )
    builder = OniomLowModelBuilder(system, _ML_ATOMS)
    cbf = builder.internal_nonbonded_force()
    assert cbf is not None
    pairs = {tuple(sorted((int(cbf.getBondParameters(i)[0]),
                           int(cbf.getBondParameters(i)[1]))))
             for i in range(cbf.getNumBonds())}
    assert (0, 1) not in pairs
    assert (1, 2) in pairs
    assert (0, 2) in pairs


# ---------------------------------------------------------------------------
# ML-MM Coulomb background tests
# ---------------------------------------------------------------------------

def test_ml_mm_coulomb_background_matches_source_ml_mm_term():
    """The ML-MM Coulomb low-model evaluates the same q_i*q_j/r sum as the
    corresponding term in the source NonbondedForce."""
    source = _build_source_system()
    positions = _positions()

    # Reference: source NonbondedForce restricted to ML-MM pairs only. We
    # build it by zeroing all LJ on ML particles, all charges on ML particles
    # except for ml-mm cross terms, and all ML-internal interactions via
    # exceptions. Cleaner: directly compute the analytic answer.
    ml_charges = [_PARAMS[i][1] for i in _ML_ATOMS]
    mm_charges = [_PARAMS[i][1] for i in _MM_ATOMS]
    coords = np.array([_PARAMS[i] for i in []])  # placeholder
    pos_np = positions.value_in_unit(unit.nanometer)
    expected = 0.0
    COULOMB = 138.935456
    for ml, qm in zip(_ML_ATOMS, ml_charges):
        for mm, qmm in zip(_MM_ATOMS, mm_charges):
            r = np.linalg.norm(pos_np[ml] - pos_np[mm])
            expected += COULOMB * qm * qmm / r

    builder = OniomLowModelBuilder(source, _ML_ATOMS)
    cnb = builder.ml_mm_coulomb_background_force(periodic=False)
    assert cnb is not None
    test_system = openmm.System()
    for _ in range(source.getNumParticles()):
        test_system.addParticle(1.0)
    test_system.addForce(cnb)
    e_test = _energy_of(test_system, positions)

    assert e_test == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_ml_mm_coulomb_background_excludes_source_exceptions():
    """An ML-MM exception in the source NonbondedForce excludes the pair
    from the ML-MM Coulomb background, so the subtraction matches what the
    host system actually contains."""
    source = _build_source_system(with_ml_mm_exception=True)
    positions = _positions()

    # Hand-compute expected: sum over (ml, mm) pairs, skipping (2, 3).
    ml_charges = [_PARAMS[i][1] for i in _ML_ATOMS]
    mm_charges = [_PARAMS[i][1] for i in _MM_ATOMS]
    pos_np = positions.value_in_unit(unit.nanometer)
    expected = 0.0
    COULOMB = 138.935456
    for ml, qm in zip(_ML_ATOMS, ml_charges):
        for mm, qmm in zip(_MM_ATOMS, mm_charges):
            if (ml, mm) == (2, 3) or (mm, ml) == (2, 3):
                continue
            r = np.linalg.norm(pos_np[ml] - pos_np[mm])
            expected += COULOMB * qm * qmm / r

    builder = OniomLowModelBuilder(source, _ML_ATOMS)
    cnb = builder.ml_mm_coulomb_background_force(periodic=False)
    test_system = openmm.System()
    for _ in range(source.getNumParticles()):
        test_system.addParticle(1.0)
    test_system.addForce(cnb)
    e_test = _energy_of(test_system, positions)
    assert e_test == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_ml_mm_coulomb_background_periodic_requires_cutoff():
    source = _build_source_system()
    builder = OniomLowModelBuilder(source, _ML_ATOMS)
    with pytest.raises(ValueError, match="cutoff"):
        builder.ml_mm_coulomb_background_force(periodic=True)


# ---------------------------------------------------------------------------
# build_all + immutability
# ---------------------------------------------------------------------------

def test_build_all_returns_expected_force_set():
    source = _build_source_system()
    builder = OniomLowModelBuilder(source, _ML_ATOMS)
    forces = builder.build_all(periodic=False)
    types = {type(f) for f in forces}
    assert openmm.HarmonicBondForce in types
    assert openmm.HarmonicAngleForce in types
    assert openmm.CustomBondForce in types
    assert openmm.CustomNonbondedForce in types


def test_builder_does_not_mutate_source():
    source = _build_source_system()
    nb = source.getForce(0)
    before_charges = [
        nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        for i in range(nb.getNumParticles())
    ]
    before_n_exc = nb.getNumExceptions()
    before_n_forces = source.getNumForces()

    builder = OniomLowModelBuilder(source, _ML_ATOMS)
    _ = builder.build_all(periodic=False)

    nb_after = source.getForce(0)
    after_charges = [
        nb_after.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        for i in range(nb_after.getNumParticles())
    ]
    assert before_charges == after_charges
    assert before_n_exc == nb_after.getNumExceptions()
    assert before_n_forces == source.getNumForces()


def test_builder_handles_system_without_nonbonded():
    """If the source has no NonbondedForce, internal_nonbonded and ml_mm
    background must return None rather than raising."""
    system = openmm.System()
    for _ in range(3):
        system.addParticle(1.0)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.15 * unit.nanometer, 100.0 * unit.kilojoule_per_mole / unit.nanometer ** 2)
    system.addForce(bonds)
    builder = OniomLowModelBuilder(system, [0, 1])
    assert builder.internal_nonbonded_force() is None
    assert builder.ml_mm_coulomb_background_force(periodic=False) is None
    bonded = builder.internal_bonded_forces()
    assert any(isinstance(f, openmm.HarmonicBondForce) for f in bonded)
