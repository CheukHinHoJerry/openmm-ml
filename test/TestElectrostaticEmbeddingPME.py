"""Unit tests for the periodic NonbondedForce / PME path of
``MLPotential.createMixedSystem(embedding='electrostatic')``.

These tests exercise the MM-side surgery only. A no-op MLPotentialImpl
is registered so the tests do not depend on MACE, torch, or GPU.

Slice 4 of docs/codex-plans/pbc-electrostatic-embedding-small-plan.md.
"""
from __future__ import annotations

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
import pytest

from openmmml import MLPotential
from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory
from openmmml.models.macepotential import MACEPotentialImpl


# ---------------------------------------------------------------------------
# Register a no-op MLPotentialImpl for tests
# ---------------------------------------------------------------------------

class _NoopImpl(MACEPotentialImpl):
    """The real electrostatic embedding with the model evaluation stubbed out.

    Electrostatic embedding is a MACE-specific embedding method, so these tests
    inherit MACEPotentialImpl.createMixedSystem() to exercise the actual
    nonbonded surgery under test.  Only addForces() is stubbed, which is what
    would otherwise require a MACE checkpoint and the PolarMACE stack; the
    surgery runs before it and is unaffected.
    """

    def __init__(self):
        super().__init__("mace", None)

    def _loadModel(self, args):
        # createMixedSystem() loads the model to check that it accepts MM
        # charges.  Stand in for a real checkpoint with an object of the class
        # it looks for, so the surgery under test runs without the PolarMACE
        # stack being installed.
        class PolarMACE:
            supports_external_electrostatics = True

            pass

        return PolarMACE(), "cpu"

    def addForces(self, topology, system, atoms, forceGroup, **args):
        return


class _NoopFactory(MLPotentialImplFactory):
    def createImpl(self, name, **args):
        return _NoopImpl()


MLPotential.registerImplFactory("noop_test_impl", _NoopFactory())


# ---------------------------------------------------------------------------
# Test system construction
# ---------------------------------------------------------------------------

# 4 atoms: ML = {0, 1}, MM = {2, 3}. Periodic 2 nm cubic box.
_PARAMS = [
    # (mass, charge_e, sigma_nm, epsilon_kj)
    (12.0, 0.6, 0.30, 0.20),   # ML 0
    (12.0, -0.4, 0.32, 0.25),  # ML 1
    (16.0, -0.8, 0.31, 0.65),  # MM 2
    (1.0, 0.4, 0.10, 0.05),    # MM 3
]
_BOX_NM = 2.0
_ML_ATOMS = [0, 1]
_MM_ATOMS = [2, 3]


def _build_periodic_system(use_pme=True, with_boundary_bonded=True, with_pre_existing_exception=False):
    system = openmm.System()
    box_vec = _BOX_NM * unit.nanometer
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(_BOX_NM, 0, 0) * unit.nanometer,
        openmm.Vec3(0, _BOX_NM, 0) * unit.nanometer,
        openmm.Vec3(0, 0, _BOX_NM) * unit.nanometer,
    )

    nonbonded = openmm.NonbondedForce()
    if use_pme:
        nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
        nonbonded.setCutoffDistance(0.6 * unit.nanometer)
    else:
        nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for mass, charge, sigma, epsilon in _PARAMS:
        system.addParticle(mass)
        nonbonded.addParticle(
            charge * unit.elementary_charge,
            sigma * unit.nanometer,
            epsilon * unit.kilojoule_per_mole,
        )

    # A pre-existing 1-3 ML-MM exception lets us check that sigma/epsilon
    # are read from the exception rather than synthesized via Lorentz-Berthelot.
    if with_pre_existing_exception:
        nonbonded.addException(
            1, 2,
            0.123 * unit.elementary_charge * unit.elementary_charge,
            0.345 * unit.nanometer,
            0.678 * unit.kilojoule_per_mole,
        )

    system.addForce(nonbonded)

    if with_boundary_bonded:
        # ML-ML bond + ML-MM bond (boundary) + MM-MM bond
        bonds = openmm.HarmonicBondForce()
        bonds.addBond(0, 1, 0.15 * unit.nanometer, 1000.0 * unit.kilojoule_per_mole / unit.nanometer ** 2)  # ML-ML
        bonds.addBond(1, 2, 0.15 * unit.nanometer, 800.0 * unit.kilojoule_per_mole / unit.nanometer ** 2)   # ML-MM (boundary)
        bonds.addBond(2, 3, 0.10 * unit.nanometer, 500.0 * unit.kilojoule_per_mole / unit.nanometer ** 2)   # MM-MM
        system.addForce(bonds)

        angles = openmm.HarmonicAngleForce()
        angles.addAngle(0, 1, 2, 1.9, 100.0 * unit.kilojoule_per_mole / unit.radian ** 2)  # ML-ML-MM (boundary)
        angles.addAngle(1, 2, 3, 1.9, 100.0 * unit.kilojoule_per_mole / unit.radian ** 2)  # ML-MM-MM (boundary)
        system.addForce(angles)

    return system


def _build_topology():
    topology = app.Topology()
    chain = topology.addChain()
    res = topology.addResidue("MOL", chain)
    elements = [app.element.carbon, app.element.carbon, app.element.oxygen, app.element.hydrogen]
    for i, el in enumerate(elements):
        topology.addAtom(f"A{i}", el, res)
    topology.setPeriodicBoxVectors(
        unit.Quantity(np.diag([_BOX_NM, _BOX_NM, _BOX_NM]), unit.nanometer)
    )
    return topology


def _get_nonbonded(system):
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            return force
    raise AssertionError("No NonbondedForce in system")


def _read_particle(force, i):
    charge, sigma, epsilon = force.getParticleParameters(i)
    return (
        charge.value_in_unit(unit.elementary_charge),
        sigma.value_in_unit(unit.nanometer),
        epsilon.value_in_unit(unit.kilojoule_per_mole),
    )


def _read_exception(force, i):
    p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(i)
    return (
        int(p1), int(p2),
        chargeProd.value_in_unit(unit.elementary_charge * unit.elementary_charge),
        sigma.value_in_unit(unit.nanometer),
        epsilon.value_in_unit(unit.kilojoule_per_mole),
    )


def _all_exceptions(force):
    return {
        tuple(sorted((p1, p2))): (cp, s, e)
        for (p1, p2, cp, s, e) in (_read_exception(force, i) for i in range(force.getNumExceptions()))
    }


def _make_mixed_system(**system_kwargs):
    system = _build_periodic_system(**system_kwargs)
    topology = _build_topology()
    potential = MLPotential("noop_test_impl")
    return potential.createMixedSystem(
        topology, system, _ML_ATOMS, embedding="electrostatic"
    ), system


# ---------------------------------------------------------------------------
# Slice 4 unit tests
# ---------------------------------------------------------------------------

def test_pme_method_preserved():
    """The PME setting on NonbondedForce must survive the surgery."""
    mixed, _ = _make_mixed_system()
    nb = _get_nonbonded(mixed)
    assert nb.getNonbondedMethod() == openmm.NonbondedForce.PME


def test_ml_charges_zeroed():
    """ML particle charges must be zero so reciprocal-space ML-* contribs vanish."""
    mixed, _ = _make_mixed_system()
    nb = _get_nonbonded(mixed)
    for i in _ML_ATOMS:
        charge, _, _ = _read_particle(nb, i)
        assert charge == pytest.approx(0.0, abs=1e-12)


def test_mm_charges_unchanged():
    """MM particle charges must be untouched by the surgery."""
    mixed, original = _make_mixed_system()
    nb_mixed = _get_nonbonded(mixed)
    nb_orig = _get_nonbonded(original)
    for i in _MM_ATOMS:
        new_charge, _, _ = _read_particle(nb_mixed, i)
        old_charge, _, _ = _read_particle(nb_orig, i)
        assert new_charge == pytest.approx(old_charge, abs=1e-12)


def test_lj_parameters_preserved_for_all_particles():
    """sigma/epsilon must be untouched (LJ stays in MM force field)."""
    mixed, original = _make_mixed_system()
    nb_mixed = _get_nonbonded(mixed)
    nb_orig = _get_nonbonded(original)
    for i in range(nb_orig.getNumParticles()):
        _, s_new, e_new = _read_particle(nb_mixed, i)
        _, s_old, e_old = _read_particle(nb_orig, i)
        assert s_new == pytest.approx(s_old, abs=1e-12)
        assert e_new == pytest.approx(e_old, abs=1e-12)


def test_ml_mm_pairs_get_no_exception():
    """ML-MM pairs must be left on the ordinary pair list.

    ML-MM Coulomb is removed by zeroing the ML particle charges, not by adding
    an exception per ML-MM pair.  An exception would also be wrong: OpenMM
    evaluates exceptions at the plain Cartesian distance rather than the
    minimum image one, so under PBC the ML-MM Lennard-Jones interaction would
    silently disappear for any pair that is only within the cutoff across a
    periodic boundary.
    """
    mixed, _ = _make_mixed_system(with_pre_existing_exception=False)
    nb = _get_nonbonded(mixed)
    excs = _all_exceptions(nb)
    for ml in _ML_ATOMS:
        for mm in _MM_ATOMS:
            key = tuple(sorted((ml, mm)))
            assert key not in excs, f"Unexpected ML-MM exception for {key}"


def test_ml_charges_zeroed_and_lj_untouched():
    """The Coulomb removal is done by zeroing the ML particle charges, which
    leaves their Lennard-Jones parameters, and so ML-MM LJ, intact."""
    mixed, _ = _make_mixed_system(with_pre_existing_exception=False)
    nb = _get_nonbonded(mixed)
    for ml in _ML_ATOMS:
        charge, sigma, epsilon = _read_particle(nb, ml)
        assert charge == pytest.approx(0.0, abs=1e-15)
        assert sigma == pytest.approx(_PARAMS[ml][2], rel=1e-12)
        assert epsilon == pytest.approx(_PARAMS[ml][3], rel=1e-12)
    for mm in _MM_ATOMS:
        charge, sigma, epsilon = _read_particle(nb, mm)
        assert charge == pytest.approx(_PARAMS[mm][1], rel=1e-12)
        assert sigma == pytest.approx(_PARAMS[mm][2], rel=1e-12)
        assert epsilon == pytest.approx(_PARAMS[mm][3], rel=1e-12)


def test_ml_mm_pre_existing_exception_keeps_lj_zeroes_charge():
    """If a pre-existing ML-MM exception had nonzero chargeProd, surgery must
    zero it but keep the explicit sigma/epsilon."""
    mixed, _ = _make_mixed_system(with_pre_existing_exception=True)
    nb = _get_nonbonded(mixed)
    excs = _all_exceptions(nb)
    cp, sigma, epsilon = excs[(1, 2)]
    assert cp == pytest.approx(0.0, abs=1e-15)
    assert sigma == pytest.approx(0.345, rel=1e-12)
    assert epsilon == pytest.approx(0.678, rel=1e-12)


def test_ml_ml_exceptions_zero_charge_and_zero_lj():
    """ML-ML pairs are fully internal to MACE — both Coulomb and LJ must be
    zeroed in the MM force field."""
    mixed, _ = _make_mixed_system()
    nb = _get_nonbonded(mixed)
    excs = _all_exceptions(nb)
    key = tuple(sorted(_ML_ATOMS))
    assert key in excs
    cp, _, epsilon = excs[key]
    assert cp == pytest.approx(0.0, abs=1e-15)
    assert epsilon == pytest.approx(0.0, abs=1e-15)


def test_mm_mm_exceptions_untouched():
    """MM-MM pairs must keep whatever the original system specified
    (here: nothing — no exceptions added by surgery)."""
    mixed, original = _make_mixed_system()
    nb_mixed = _get_nonbonded(mixed)
    nb_orig = _get_nonbonded(original)
    excs_orig = _all_exceptions(nb_orig)
    excs_mixed = _all_exceptions(nb_mixed)
    mm_key = tuple(sorted(_MM_ATOMS))
    assert excs_mixed.get(mm_key) == excs_orig.get(mm_key)


def test_boundary_bonded_terms_preserved():
    """Boundary bonded terms (HarmonicBond/Angle that connect ML-MM) must
    survive — they remain classical in electrostatic embedding."""
    mixed, _ = _make_mixed_system(with_boundary_bonded=True)
    bond_force = next(f for f in mixed.getForces() if isinstance(f, openmm.HarmonicBondForce))
    bond_pairs = set()
    for i in range(bond_force.getNumBonds()):
        p1, p2, _, _ = bond_force.getBondParameters(i)
        bond_pairs.add(tuple(sorted((int(p1), int(p2)))))
    # ML-MM boundary bond and MM-MM bond must remain.
    assert (1, 2) in bond_pairs
    assert (2, 3) in bond_pairs
    # ML-internal bond must be removed.
    assert (0, 1) not in bond_pairs

    angle_force = next(f for f in mixed.getForces() if isinstance(f, openmm.HarmonicAngleForce))
    angle_triples = set()
    for i in range(angle_force.getNumAngles()):
        p1, p2, p3, _, _ = angle_force.getAngleParameters(i)
        angle_triples.add((int(p1), int(p2), int(p3)))
    # Boundary angles (touching at least one MM atom) preserved.
    assert (0, 1, 2) in angle_triples
    assert (1, 2, 3) in angle_triples


def test_reciprocal_space_pme_matches_mm_only_reference():
    """PME reciprocal-space contribution from the mixed-system NonbondedForce
    must equal that of an MM-only reference (where ML charges are explicitly
    zeroed but the system is otherwise identical)."""
    mixed, original = _make_mixed_system(with_boundary_bonded=False)

    # Strip the bonded forces so only NonbondedForce contributes (we are
    # comparing the reciprocal-space + direct-space Coulomb/LJ summed term).
    def _strip_to_nonbonded(system):
        # Remove non-NonbondedForce forces by index, descending.
        for idx in reversed(range(system.getNumForces())):
            if not isinstance(system.getForce(idx), openmm.NonbondedForce):
                system.removeForce(idx)
        return system

    _strip_to_nonbonded(mixed)

    # Build MM-only reference by zeroing ML charges in a fresh original system.
    mm_only = _build_periodic_system(with_boundary_bonded=False)
    nb_ref = _get_nonbonded(mm_only)
    for i in _ML_ATOMS:
        _, sigma, epsilon = nb_ref.getParticleParameters(i)
        nb_ref.setParticleParameters(i, 0.0 * unit.elementary_charge, sigma, epsilon)
    # Add the same exception structure (LB ML-MM zero-charge + ML-ML LJ-zero)
    # so that direct-space exclusion subtractions match between the two systems.
    # That is what the surgery itself produced; copy it onto the reference.
    nb_mixed = _get_nonbonded(mixed)
    existing_pairs = set(
        tuple(sorted((int(p1), int(p2))))
        for i in range(nb_ref.getNumExceptions())
        for (p1, p2, *_rest) in [nb_ref.getExceptionParameters(i)]
    )
    for i in range(nb_mixed.getNumExceptions()):
        p1, p2, cp, sigma, epsilon = nb_mixed.getExceptionParameters(i)
        if tuple(sorted((int(p1), int(p2)))) in existing_pairs:
            continue
        nb_ref.addException(int(p1), int(p2), cp, sigma, epsilon)
    _strip_to_nonbonded(mm_only)

    positions = [
        openmm.Vec3(0.20, 0.30, 0.40),
        openmm.Vec3(0.45, 0.30, 0.40),
        openmm.Vec3(0.70, 0.30, 0.40),
        openmm.Vec3(0.80, 0.30, 0.40),
    ] * unit.nanometer

    platform = openmm.Platform.getPlatformByName("Reference")
    ctx_mixed = openmm.Context(mixed, openmm.VerletIntegrator(0.001), platform)
    ctx_ref = openmm.Context(mm_only, openmm.VerletIntegrator(0.001), platform)
    ctx_mixed.setPositions(positions)
    ctx_ref.setPositions(positions)

    e_mixed = ctx_mixed.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    e_ref = ctx_ref.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    assert e_mixed == pytest.approx(e_ref, rel=1e-6, abs=1e-6)


# ---------------------------------------------------------------------------
# Electrostatics the surgery cannot account for.
#
# The ML-MM Coulomb is removed from the force field on the understanding that
# the model supplies it. Any Coulomb term this method cannot find is either left
# in place and counted twice, or removed and never replaced. Both give a wrong
# energy with no error, so these cases are refused rather than guessed at.
# ---------------------------------------------------------------------------


def _minimal_topology(numParticles):
    from openmm.app import element

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("X", chain)
    for i in range(numParticles):
        topology.addAtom(f"H{i}", element.hydrogen, residue)
    return topology


def _createMixedSystem(system, **args):
    impl = _NoopImpl()
    return impl.createMixedSystem(
        _minimal_topology(system.getNumParticles()), system, _ML_ATOMS, 0, False,
        "electrostatic", **args,
    )


def test_multiple_nonbonded_forces_rejected():
    """The MM charges given to the model come from one NonbondedForce, so more
    than one is ambiguous: the surgery would zero ML charges in all of them
    while the model saw the charges of only the first."""
    system = _build_periodic_system()
    system.addForce(openmm.NonbondedForce())
    with pytest.raises(ValueError, match="Multiple NonbondedForce"):
        _createMixedSystem(system)


def _addCoulombCustomNonbondedForce(system):
    force = openmm.CustomNonbondedForce("138.935456*q1*q2/r")
    force.addPerParticleParameter("q")
    for index in range(system.getNumParticles()):
        force.addParticle([_PARAMS[index][1]])
    system.addForce(force)
    return system


def test_custom_nonbonded_force_requires_an_answer():
    """A CustomNonbondedForce's energy expression is arbitrary, so whether it
    carries electrostatics cannot be determined here and must be declared."""
    system = _addCoulombCustomNonbondedForce(_build_periodic_system())
    with pytest.raises(ValueError, match="unknown whether it includes electrostatic"):
        _createMixedSystem(system)


def test_custom_nonbonded_force_with_charges_needs_the_parameter_name():
    """Declaring that it does carry electrostatics is not enough on its own:
    the charge cannot be zeroed without knowing which parameter holds it."""
    system = _addCoulombCustomNonbondedForce(_build_periodic_system())
    with pytest.raises(ValueError, match="must name the per-particle parameter"):
        _createMixedSystem(system, customNonbondedHasCharges=True)


def test_custom_nonbonded_force_unknown_parameter_name_rejected():
    """Naming a parameter the force does not define is an error, not a no-op
    that would leave the electrostatics in place."""
    system = _addCoulombCustomNonbondedForce(_build_periodic_system())
    with pytest.raises(ValueError, match="no per-particle parameter"):
        _createMixedSystem(system, customNonbondedHasCharges=True,
                           customNonbondedChargeParameter="charge")


def test_custom_nonbonded_charge_parameter_zeroed_on_ml_atoms():
    """Naming the charge parameter zeroes it on the ML atoms, which removes the
    ML-MM Coulomb the custom force would otherwise still contribute."""
    system = _addCoulombCustomNonbondedForce(_build_periodic_system())
    mixed = _createMixedSystem(system, customNonbondedHasCharges=True,
                               customNonbondedChargeParameter="q")
    custom = next(f for f in mixed.getForces() if isinstance(f, openmm.CustomNonbondedForce))
    for atom in _ML_ATOMS:
        assert custom.getParticleParameters(atom)[0] == pytest.approx(0.0, abs=1e-12)
    for atom in _MM_ATOMS:
        assert custom.getParticleParameters(atom)[0] == pytest.approx(_PARAMS[atom][1], rel=1e-12)


def test_custom_nonbonded_force_without_charges_accepted():
    """Declaring it carries none proceeds, with ML-ML excluded as usual."""
    system = _addCoulombCustomNonbondedForce(_build_periodic_system())
    mixed = _createMixedSystem(system, customNonbondedHasCharges=False)
    custom = next(f for f in mixed.getForces() if isinstance(f, openmm.CustomNonbondedForce))
    exclusions = {
        tuple(sorted(custom.getExclusionParticles(i)))
        for i in range(custom.getNumExclusions())
    }
    assert tuple(sorted(_ML_ATOMS)) in exclusions


def test_custom_nonbonded_charges_reach_the_model():
    """When the electrostatics live in the CustomNonbondedForce, the charges the
    model is given must come from there too.

    Reading them from the NonbondedForce in that case hands the model zeros
    while the surgery has already removed the real ML-MM Coulomb, so the
    interaction disappears rather than being computed by the model.
    """
    from openmmml.models.macepotential import _prepareMMEmbedding

    system = openmm.System()
    for _ in range(4):
        system.addParticle(1.0)
    lj = openmm.NonbondedForce()          # Lennard-Jones only, no charges
    for _, _, sigma, epsilon in _PARAMS:
        lj.addParticle(0.0, sigma, epsilon)
    system.addForce(lj)
    coulomb = openmm.CustomNonbondedForce("138.935456*q1*q2/r")
    coulomb.addPerParticleParameter("q")
    for _, charge, _, _ in _PARAMS:
        coulomb.addParticle([charge])
    system.addForce(coulomb)

    expected = [_PARAMS[i][1] for i in _MM_ATOMS]
    charges = _prepareMMEmbedding(system, _ML_ATOMS, "q")["mm_charges"]
    np.testing.assert_allclose(charges, expected, atol=1e-12)
