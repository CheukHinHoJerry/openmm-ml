"""End-to-end tests for PolarMACE electrostatic embedding via openmmml.

Covers both:

* Non-periodic plumbing: MM positions/charges flow into PolarMACE,
  electrostatic embedding shifts the ML energy and creates a back-reaction
  force on the MM atoms (Slice 4 cross-check, fills the gap that the only
  existing electrostatic-embedding tests in TestMACEPotential.py exercise
  the helpers in isolation rather than the full plumbing).

* PBC: ``MLPotential('mace', modelPath=...).createMixedSystem(
  embedding='electrostatic')`` is translation-invariant under PBC --
  shifting all positions by one full box vector must leave the potential
  energy and per-atom forces unchanged. (Slice 5 of
  docs/codex-plans/pbc-electrostatic-embedding-small-plan.md.)

The tests train nothing -- a tiny random PolarMACE is built and serialized
in a module-scoped fixture so the suite runs in seconds on CPU.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
import pytest
import torch

torch.serialization.add_safe_globals([slice])

mace = pytest.importorskip("mace", reason="mace is not installed")
from e3nn import o3  # noqa: E402

from openmmml import MLPotential  # noqa: E402
from openmmml.models import macepotential  # noqa: E402
from mace.modules import interaction_classes  # noqa: E402
from mace.modules.extensions import PolarMACE  # noqa: E402


_BOX_NM = 1.4   # 14 Angstrom box (large enough for r_max=4 A and PME)
_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# PolarMACE model + save to tempfile
# ---------------------------------------------------------------------------

def _build_polar_mace(device: torch.device, dtype: torch.dtype) -> PolarMACE:
    fixedpoint_update_config = {
        "type": "AgnosticEmbeddedOneBodyVariableUpdate",
        "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding",
        "nonlinearity_cls": "MLPNonLinearity",
    }
    field_readout_config = {"type": "OneBodyMLPFieldReadout"}
    return PolarMACE(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=interaction_classes[
            "RealAgnosticResidualNonLinearInteractionBlock"
        ],
        interaction_cls_first=interaction_classes[
            "RealAgnosticResidualNonLinearInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("4x0e + 4x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=torch.zeros(2, dtype=dtype, device=device),
        avg_num_neighbors=3.0,
        atomic_numbers=[1, 8],
        correlation=1,
        gate=torch.nn.functional.silu,
        radial_MLP=[16, 16],
        radial_type="bessel",
        kspace_cutoff_factor=1.0,
        atomic_multipoles_max_l=1,
        atomic_multipoles_smearing_width=1.0,
        field_feature_max_l=1,
        field_feature_widths=[1.0],
        field_feature_norms=[1.0, 1.0],
        num_recursion_steps=1,
        field_si=False,
        include_electrostatic_self_interaction=False,
        add_local_electron_energy=True,
        field_dependence_type="AgnosticEmbeddedOneBodyVariableUpdate",
        final_field_readout_type="OneBodyMLPFieldReadout",
        return_electrostatic_potentials=False,
        heads=["Default"],
        field_norm_factor=1.0,
        fixedpoint_update_config=fixedpoint_update_config,
        field_readout_config=field_readout_config,
    ).to(device=device, dtype=dtype)


@pytest.fixture(scope="module")
def polar_mace_model_path():
    torch.manual_seed(7)
    model = _build_polar_mace(torch.device("cpu"), _DTYPE)
    model.eval()
    tmpdir = tempfile.mkdtemp(prefix="polar_mace_pbc_test_")
    path = os.path.join(tmpdir, "polar_mace_test.pt")
    torch.save(model, path)
    yield path
    try:
        os.remove(path)
        os.rmdir(tmpdir)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Periodic OpenMM topology + system: 1 ML water + N MM waters
# ---------------------------------------------------------------------------

# TIP3P-ish charges and LJ for the test (units consistent with OpenMM defaults).
_O_CHARGE = -0.834
_H_CHARGE = 0.417
_O_SIGMA_NM = 0.31507
_O_EPS_KJ = 0.6364
_H_SIGMA_NM = 1.0e-3
_H_EPS_KJ = 0.0


def _add_water(system, nonbonded, bonds, angles, masses_charges_lj):
    """Append a 3-atom water to system+forces; returns particle indices."""
    indices = []
    for mass, charge, sigma, epsilon in masses_charges_lj:
        idx = system.addParticle(mass)
        nonbonded.addParticle(
            charge * unit.elementary_charge,
            sigma * unit.nanometer,
            epsilon * unit.kilojoule_per_mole,
        )
        indices.append(idx)
    o, h1, h2 = indices
    bonds.addBond(o, h1, 0.09572 * unit.nanometer, 4.5e5 * unit.kilojoule_per_mole / unit.nanometer ** 2)
    bonds.addBond(o, h2, 0.09572 * unit.nanometer, 4.5e5 * unit.kilojoule_per_mole / unit.nanometer ** 2)
    angles.addAngle(h1, o, h2, 1.824, 460.0 * unit.kilojoule_per_mole / unit.radian ** 2)
    nonbonded.addException(
        o, h1,
        0.0 * unit.elementary_charge * unit.elementary_charge,
        0.5 * (_O_SIGMA_NM + _H_SIGMA_NM) * unit.nanometer,
        0.0 * unit.kilojoule_per_mole,
    )
    nonbonded.addException(
        o, h2,
        0.0 * unit.elementary_charge * unit.elementary_charge,
        0.5 * (_O_SIGMA_NM + _H_SIGMA_NM) * unit.nanometer,
        0.0 * unit.kilojoule_per_mole,
    )
    nonbonded.addException(
        h1, h2,
        0.0 * unit.elementary_charge * unit.elementary_charge,
        _H_SIGMA_NM * unit.nanometer,
        0.0 * unit.kilojoule_per_mole,
    )
    return indices


def _build_topology_and_system(num_mm_waters=3, periodic=True, mm_oxygen_charge=_O_CHARGE):
    topology = app.Topology()
    chain = topology.addChain()

    # ML water (residue 0).
    res = topology.addResidue("HOH", chain)
    a_o = topology.addAtom("O", app.element.oxygen, res)
    a_h1 = topology.addAtom("H1", app.element.hydrogen, res)
    a_h2 = topology.addAtom("H2", app.element.hydrogen, res)
    topology.addBond(a_o, a_h1)
    topology.addBond(a_o, a_h2)

    for _ in range(num_mm_waters):
        res = topology.addResidue("HOH", chain)
        a_o = topology.addAtom("O", app.element.oxygen, res)
        a_h1 = topology.addAtom("H1", app.element.hydrogen, res)
        a_h2 = topology.addAtom("H2", app.element.hydrogen, res)
        topology.addBond(a_o, a_h1)
        topology.addBond(a_o, a_h2)

    if periodic:
        topology.setPeriodicBoxVectors(
            unit.Quantity(np.diag([_BOX_NM, _BOX_NM, _BOX_NM]), unit.nanometer)
        )

    system = openmm.System()
    if periodic:
        system.setDefaultPeriodicBoxVectors(
            openmm.Vec3(_BOX_NM, 0, 0) * unit.nanometer,
            openmm.Vec3(0, _BOX_NM, 0) * unit.nanometer,
            openmm.Vec3(0, 0, _BOX_NM) * unit.nanometer,
        )
    nonbonded = openmm.NonbondedForce()
    if periodic:
        nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
        nonbonded.setCutoffDistance(0.5 * unit.nanometer)
    else:
        nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    bonds = openmm.HarmonicBondForce()
    angles = openmm.HarmonicAngleForce()

    ml_o = (15.999, _O_CHARGE, _O_SIGMA_NM, _O_EPS_KJ)
    ml_h = (1.008, _H_CHARGE, _H_SIGMA_NM, _H_EPS_KJ)
    mm_o = (15.999, mm_oxygen_charge, _O_SIGMA_NM, _O_EPS_KJ)
    mm_h_charge = -mm_oxygen_charge / 2.0
    mm_h = (1.008, mm_h_charge, _H_SIGMA_NM, _H_EPS_KJ)

    _add_water(system, nonbonded, bonds, angles, [ml_o, ml_h, ml_h])
    for _ in range(num_mm_waters):
        _add_water(system, nonbonded, bonds, angles, [mm_o, mm_h, mm_h])

    system.addForce(nonbonded)
    system.addForce(bonds)
    system.addForce(angles)

    return topology, system


def _initial_positions(num_mm_waters):
    """Return positions in nanometers. ML water at ~box centre, MM waters scattered."""
    base_o = np.array([
        [0.55, 0.55, 0.55],   # ML O
        [0.20, 0.20, 0.20],
        [0.95, 0.20, 0.30],
        [0.30, 0.95, 0.95],
        [0.95, 0.95, 0.95],
        [0.10, 0.50, 0.10],
    ])[: 1 + num_mm_waters]
    H1_OFFSET = np.array([0.09572, 0.0, 0.0])
    H2_OFFSET = np.array([-0.0240, 0.0927, 0.0])  # 104.5 deg, 0.09572 nm bond
    coords = []
    for o in base_o:
        coords.append(o)
        coords.append(o + H1_OFFSET)
        coords.append(o + H2_OFFSET)
    return np.array(coords) * unit.nanometer


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def _energy_and_forces(context):
    state = context.getState(getEnergy=True, getForces=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    return float(e), np.asarray(f)


def test_polar_mace_pbc_runs_and_is_finite(polar_mace_model_path):
    """Smoke test: createMixedSystem(embedding='electrostatic') with PolarMACE
    under PBC returns a finite energy."""
    topology, mm_system = _build_topology_and_system(num_mm_waters=2)
    potential = MLPotential("mace", modelPath=polar_mace_model_path)
    mixed_system = potential.createMixedSystem(
        topology, mm_system, [0, 1, 2], embedding="electrostatic"
    )
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(mixed_system, openmm.VerletIntegrator(0.001), platform)
    context.setPositions(_initial_positions(num_mm_waters=2))
    e, f = _energy_and_forces(context)
    assert np.isfinite(e)
    assert np.all(np.isfinite(f))


def test_polar_mace_pbc_translation_invariance(polar_mace_model_path):
    """Energy and forces must be invariant under whole-system translation by
    one full box vector — the fundamental PBC sanity check."""
    topology, mm_system = _build_topology_and_system(num_mm_waters=2)
    potential = MLPotential("mace", modelPath=polar_mace_model_path)
    mixed_system = potential.createMixedSystem(
        topology, mm_system, [0, 1, 2], embedding="electrostatic"
    )
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(mixed_system, openmm.VerletIntegrator(0.001), platform)

    pos = _initial_positions(num_mm_waters=2)
    context.setPositions(pos)
    e0, f0 = _energy_and_forces(context)

    # Translate all atoms by +x box length. Forces are reported per-atom in the
    # same order, so direct comparison is valid.
    pos_shifted = (
        pos.value_in_unit(unit.nanometer) + np.array([_BOX_NM, 0.0, 0.0])
    ) * unit.nanometer
    context.setPositions(pos_shifted)
    e1, f1 = _energy_and_forces(context)

    assert e1 == pytest.approx(e0, rel=1e-6, abs=1e-4)
    np.testing.assert_allclose(f1, f0, rtol=1e-5, atol=1e-3)


def test_polar_mace_pbc_ml_atom_near_boundary(polar_mace_model_path):
    """If an ML atom sits near a periodic boundary, its energy must match the
    energy from the unwrapped image (translation by one full box must give
    the same result)."""
    topology, mm_system = _build_topology_and_system(num_mm_waters=2)
    potential = MLPotential("mace", modelPath=polar_mace_model_path)
    mixed_system = potential.createMixedSystem(
        topology, mm_system, [0, 1, 2], embedding="electrostatic"
    )
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(mixed_system, openmm.VerletIntegrator(0.001), platform)

    # Put the ML water O near x=0 (boundary) with H atoms straddling the box.
    pos_np = _initial_positions(num_mm_waters=2).value_in_unit(unit.nanometer)
    pos_np[0] = np.array([0.05, 0.55, 0.55])           # O near +x boundary edge
    pos_np[1] = pos_np[0] + np.array([0.09572, 0.0, 0.0])  # H still inside box
    pos_np[2] = pos_np[0] + np.array([-0.024, 0.0927, 0.0])  # H near boundary
    context.setPositions(pos_np * unit.nanometer)
    e_near, f_near = _energy_and_forces(context)
    assert np.isfinite(e_near)
    assert np.all(np.isfinite(f_near))

    # Translate everything by one full box: identical energy and forces.
    context.setPositions((pos_np + np.array([_BOX_NM, 0.0, 0.0])) * unit.nanometer)
    e_shift, f_shift = _energy_and_forces(context)
    assert e_shift == pytest.approx(e_near, rel=1e-6, abs=1e-4)
    np.testing.assert_allclose(f_shift, f_near, rtol=1e-5, atol=1e-3)


# ---------------------------------------------------------------------------
# Non-PBC tests: MACE electrostatic plumbing through openmmml
# ---------------------------------------------------------------------------

def _build_nonpbc_mixed(potential, num_mm_waters, embedding, mm_oxygen_charge=_O_CHARGE):
    topology, mm_system = _build_topology_and_system(
        num_mm_waters=num_mm_waters, periodic=False, mm_oxygen_charge=mm_oxygen_charge
    )
    mixed_system = potential.createMixedSystem(
        topology, mm_system, [0, 1, 2], embedding=embedding
    )
    platform = openmm.Platform.getPlatformByName("Reference")
    return mixed_system, openmm.Context(mixed_system, openmm.VerletIntegrator(0.001), platform)


def _nonpbc_initial_positions(num_mm_waters):
    base_o = np.array([
        [0.0, 0.0, 0.0],     # ML O at origin
        [0.50, 0.0, 0.0],    # MM water on +x
        [0.0, 0.50, 0.0],    # MM water on +y
        [0.0, 0.0, 0.50],    # MM water on +z
    ])[: 1 + num_mm_waters]
    H1_OFFSET = np.array([0.09572, 0.0, 0.0])
    H2_OFFSET = np.array([-0.0240, 0.0927, 0.0])
    coords = []
    for o in base_o:
        coords.append(o)
        coords.append(o + H1_OFFSET)
        coords.append(o + H2_OFFSET)
    return np.array(coords) * unit.nanometer


def test_polar_mace_electrostatic_nonpbc_smoke(polar_mace_model_path):
    """Non-PBC: createMixedSystem(embedding='electrostatic') with PolarMACE
    runs end-to-end and produces finite energy/forces."""
    potential = MLPotential("mace", modelPath=polar_mace_model_path)
    _, ctx = _build_nonpbc_mixed(potential, num_mm_waters=2, embedding="electrostatic")
    ctx.setPositions(_nonpbc_initial_positions(num_mm_waters=2))
    e, f = _energy_and_forces(ctx)
    assert np.isfinite(e)
    assert np.all(np.isfinite(f))


def test_polar_mace_electrostatic_changes_energy_vs_mechanical(polar_mace_model_path):
    """Mechanical embedding does NOT pass MM charges into MACE; electrostatic
    does. With nonzero MM charges, the two modes must give different ML energies
    -- the only path that exercises the openmmml -> mm_charges plumbing."""
    potential = MLPotential("mace", modelPath=polar_mace_model_path)
    positions = _nonpbc_initial_positions(num_mm_waters=2)

    _, ctx_mech = _build_nonpbc_mixed(potential, num_mm_waters=2, embedding="mechanical")
    ctx_mech.setPositions(positions)
    e_mech, _ = _energy_and_forces(ctx_mech)

    _, ctx_elec = _build_nonpbc_mixed(potential, num_mm_waters=2, embedding="electrostatic")
    ctx_elec.setPositions(positions)
    e_elec, _ = _energy_and_forces(ctx_elec)

    # Energies must not coincide -- with TIP3P-ish MM charges there is a real
    # ML-MM Coulomb piece that mechanical embedding doesn't see.
    assert abs(e_elec - e_mech) > 1e-3


def test_polar_mace_electrostatic_zero_mm_charges_matches_mechanical(polar_mace_model_path):
    """If MM charges are zero, the ML-MM Coulomb term vanishes and electrostatic
    embedding should reproduce mechanical embedding (within float64 round-off)."""
    potential = MLPotential("mace", modelPath=polar_mace_model_path)
    positions = _nonpbc_initial_positions(num_mm_waters=2)

    _, ctx_mech = _build_nonpbc_mixed(
        potential, num_mm_waters=2, embedding="mechanical", mm_oxygen_charge=0.0
    )
    ctx_mech.setPositions(positions)
    e_mech, f_mech = _energy_and_forces(ctx_mech)

    _, ctx_elec = _build_nonpbc_mixed(
        potential, num_mm_waters=2, embedding="electrostatic", mm_oxygen_charge=0.0
    )
    ctx_elec.setPositions(positions)
    e_elec, f_elec = _energy_and_forces(ctx_elec)

    assert e_elec == pytest.approx(e_mech, rel=1e-5, abs=1e-3)
    np.testing.assert_allclose(f_elec, f_mech, rtol=1e-4, atol=1e-2)


def test_polar_mace_electrostatic_mm_atoms_receive_back_reaction(polar_mace_model_path):
    """The MACE mm_forces path must scatter forces onto MM atoms. With nonzero
    MM charges, MM atoms must feel a non-trivial force; with zero MM charges
    they must not (modulo MM-internal forces in the unmodified pieces of the
    system, which we factor out by subtraction)."""
    potential = MLPotential("mace", modelPath=polar_mace_model_path)
    positions = _nonpbc_initial_positions(num_mm_waters=2)

    _, ctx_charged = _build_nonpbc_mixed(
        potential, num_mm_waters=2, embedding="electrostatic", mm_oxygen_charge=_O_CHARGE
    )
    ctx_charged.setPositions(positions)
    _, f_charged = _energy_and_forces(ctx_charged)

    _, ctx_neutral = _build_nonpbc_mixed(
        potential, num_mm_waters=2, embedding="electrostatic", mm_oxygen_charge=0.0
    )
    ctx_neutral.setPositions(positions)
    _, f_neutral = _energy_and_forces(ctx_neutral)

    # MM atoms are indices 3..8 (2 waters); their force differs between charged
    # and neutral runs, and that difference is exactly the ML-MM back reaction
    # routed through openmmml.
    delta_mm = f_charged[3:] - f_neutral[3:]
    assert np.linalg.norm(delta_mm) > 1e-2


def test_polar_mace_electrostatic_mm_charge_displacement_changes_ml_force(
    polar_mace_model_path,
):
    """Moving an MM charge changes the field on the ML region and therefore
    the force on the ML atoms. This is the cleanest end-to-end check that the
    field embedding is wired through openmmml into MACE."""
    potential = MLPotential("mace", modelPath=polar_mace_model_path)

    pos_a = _nonpbc_initial_positions(num_mm_waters=2)
    # Move the second MM water (atoms 6, 7, 8) by +0.05 nm in x.
    pos_b_np = pos_a.value_in_unit(unit.nanometer).copy()
    pos_b_np[6:9, 0] += 0.05
    pos_b = pos_b_np * unit.nanometer

    _, ctx = _build_nonpbc_mixed(potential, num_mm_waters=2, embedding="electrostatic")
    ctx.setPositions(pos_a)
    _, f_a = _energy_and_forces(ctx)
    ctx.setPositions(pos_b)
    _, f_b = _energy_and_forces(ctx)

    # ML forces (atoms 0..2) must have changed: the ML region polarization
    # responds to the shifted MM field.
    delta_ml = f_b[:3] - f_a[:3]
    assert np.linalg.norm(delta_ml) > 1e-3


# ---------------------------------------------------------------------------
# Link-atom CHARGE redistribution (Z1 / DZ1) — end-to-end via MACE
# ---------------------------------------------------------------------------
#
# The unit tests in TestLinkChargeRedistribution.py cover the pure helper
# (apply_link_charge_redistribution). The tests below exercise the actual
# integration into MACEPotentialImpl.addForces and verify that:
#   (a) the scheme parameter propagates all the way to MACE's mm_charges
#       input (different schemes give different energies/forces);
#   (b) forces remain conservative under each scheme (F = -dE/dx, FD-tested).


def _build_capped_nonpbc_chain():
    """Tiny 6-atom chain: ML region {0,1,2} bonded to MM atom 3 (the M atom)
    which is bonded to MM atoms {4,5} (M's neighbours, for DZ1 to populate).
    All MM atoms carry non-zero partial charges so the linkChargeScheme has
    something to redistribute.
    """
    system = openmm.System()
    # Make all six atoms hydrogen (element 1) so MACE's atomic-number map
    # exercised by the polar_mace_model_path fixture works (it knows H+O only).
    for _ in range(6):
        system.addParticle(1.0)
    nb = openmm.NonbondedForce()
    # ML atoms get zero charge (they'll be re-zeroed by createMixedSystem anyway,
    # but make it explicit). MM atoms get something the scheme can act on.
    nb.addParticle(0.0,  0.30, 0.20)   # 0  ML
    nb.addParticle(0.0,  0.30, 0.20)   # 1  ML
    nb.addParticle(0.0,  0.30, 0.20)   # 2  ML  (Q atom in the cut)
    nb.addParticle(-0.40, 0.30, 0.20)  # 3  MM  (the M atom)
    nb.addParticle(0.15, 0.30, 0.20)   # 4  MM  (M1 neighbour)
    nb.addParticle(0.25, 0.30, 0.20)   # 5  MM  (M1 neighbour)
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    system.addForce(nb)

    topology = app.Topology()
    chain = topology.addChain()
    res = topology.addResidue("X", chain)
    atoms = [topology.addAtom(f"H{i}", elem.hydrogen, res) for i in range(6)]
    # Bonds: 0-1, 1-2, 2-3 (Q-M cut), 3-4 (M-M1), 3-5 (M-M1)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)]:
        topology.addBond(atoms[a], atoms[b])
    return topology, system


def _capped_chain_positions():
    return np.array([
        [0.00, 0.0, 0.0],
        [0.11, 0.0, 0.0],
        [0.22, 0.0, 0.0],
        [0.33, 0.0, 0.0],   # M atom
        [0.40, 0.07, 0.0],  # M1
        [0.40, -0.07, 0.0], # M1
    ]) * unit.nanometer


def _build_capped_mixed(potential, scheme):
    topology, mm_system = _build_capped_nonpbc_chain()
    mixed = potential.createMixedSystem(
        topology, mm_system, atoms=[0, 1, 2],
        embedding="electrostatic",
        linkRecords=[(2, 3, 1.09)],   # Q=2 (in ML), M=3 (MM)
        linkChargeScheme=scheme,
    )
    plat = openmm.Platform.getPlatformByName("Reference")
    ctx = openmm.Context(mixed, openmm.VerletIntegrator(0.001), plat)
    ctx.setPositions(_capped_chain_positions())
    return mixed, ctx


# elem import for the topology builder; sits with the openmm.app namespace.
from openmm.app import element as elem  # noqa: E402  (kept near point of use)


def _mmChargesReachingModel(potential, scheme):
    """The mm_charges array that createMixedSystem bakes into the ML force.

    Asserting on the array rather than on the energy it produces is deliberate.
    The energy route would only distinguish the schemes if the model in use
    responded to the MM charge *values*, and the synthetic checkpoint built by
    the polar_mace_model_path fixture does not for this fixture's geometry: its
    energy is bit-identical whether mm_charges are passed unchanged, zeroed, or
    scaled tenfold.  What the redistribution schemes are responsible for is the
    contents of that array, so that is what these tests pin, exactly rather
    than as an inequality.
    """
    captured = {}
    original = macepotential._computeMACE

    def capture(state, **args):
        if args.get("mmInfo") is not None and "mm_charges" not in captured:
            captured["mm_atoms"] = np.asarray(args["mmInfo"]["mm_atoms"])
            captured["mm_charges"] = np.asarray(args["mmInfo"]["mm_charges"])
        return original(state, **args)

    macepotential._computeMACE = capture
    try:
        _, context = _build_capped_mixed(potential, scheme=scheme)
        context.getState(getEnergy=True)
    finally:
        macepotential._computeMACE = original
    assert "mm_charges" in captured, "the ML force never received MM charges"
    return captured["mm_atoms"], captured["mm_charges"]


def test_link_charge_scheme_none_passes_forcefield_charges(polar_mace_model_path):
    """With scheme='none' the model sees the MM force field charges verbatim."""
    potential = MLPotential("mace", modelPath=polar_mace_model_path)
    mmAtoms, mmCharges = _mmChargesReachingModel(potential, "none")
    # Atoms 3 (the M atom), 4 and 5 (its MM neighbours) from
    # _build_capped_nonpbc_chain.
    np.testing.assert_array_equal(mmAtoms, [3, 4, 5])
    np.testing.assert_allclose(mmCharges, [-0.40, 0.15, 0.25], atol=1e-12)


def test_link_charge_scheme_z1_zeros_the_m_atom(polar_mace_model_path):
    """Z1 zeros the M-atom charge and leaves every other MM charge alone,
    which changes the total MM charge by -q_M."""
    potential = MLPotential("mace", modelPath=polar_mace_model_path)
    _, mmCharges = _mmChargesReachingModel(potential, "z1")
    np.testing.assert_allclose(mmCharges, [0.0, 0.15, 0.25], atol=1e-12)
    assert mmCharges.sum() == pytest.approx(0.40, abs=1e-12)


def test_link_charge_scheme_dz1_spreads_the_m_charge_and_conserves(polar_mace_model_path):
    """DZ1 zeros the M-atom charge and spreads it over M's two MM neighbours,
    -0.40/2 = -0.20 each, leaving the total MM charge unchanged."""
    potential = MLPotential("mace", modelPath=polar_mace_model_path)
    _, mmCharges = _mmChargesReachingModel(potential, "dz1")
    np.testing.assert_allclose(mmCharges, [0.0, 0.15 - 0.20, 0.25 - 0.20], atol=1e-12)
    assert mmCharges.sum() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("scheme", ["none", "z1", "dz1"])
def test_link_charge_scheme_fd_force_consistency(polar_mace_model_path, scheme):
    """For each scheme, the MACE-side forces must equal the negative gradient
    of the MACE-side energy. Central finite difference on two QM atoms in f64.

    Tolerance is intentionally loose (10 kJ/mol/nm): this runs against a
    synthetic *untrained* MACE checkpoint whose local PES curvature is
    arbitrary, so the central-difference truncation term is much larger than
    on a trained model. The threshold is sized to catch a real
    non-conservation regression (the `returnEnergyType='interaction_energy'`
    bug fixed in PR #16 reproducibly showed |dF| in the hundreds of
    kJ/mol/nm) while tolerating that synthetic-model noise floor. Do not
    tighten without re-running against a trained model first.
    """
    potential = MLPotential("mace", modelPath=polar_mace_model_path)
    _, ctx = _build_capped_mixed(potential, scheme=scheme)

    pos0 = _capped_chain_positions().value_in_unit(unit.nanometer).copy()
    delta = 5e-5  # nm  (= 5e-4 Å, near the FD valley minimum for synthetic models)

    ctx.setPositions(pos0 * unit.nanometer)
    e_ref = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)
    f_ref = ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer)

    max_abs_err = 0.0
    for ia in (0, 1):           # two QM atoms
        for ax in range(3):
            saved = pos0[ia, ax]
            pos0[ia, ax] = saved + delta
            ctx.setPositions(pos0 * unit.nanometer)
            e_p = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole)
            pos0[ia, ax] = saved - delta
            ctx.setPositions(pos0 * unit.nanometer)
            e_m = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole)
            pos0[ia, ax] = saved
            f_num = -(e_p - e_m) / (2.0 * delta)
            err = abs(f_num - f_ref[ia, ax])
            max_abs_err = max(max_abs_err, err)
    assert max_abs_err < 10.0, (
        f"FD vs analytical force mismatch for scheme='{scheme}': "
        f"max |dF| = {max_abs_err:.3e} kJ/mol/nm (expected < 10)"
    )
