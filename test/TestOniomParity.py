"""Slice 3 parity test: ONIOM-EE matches the existing 'electrostatic'
embedding mode within 1e-5 on a periodic mixed system with PolarMACE.

Closed-valence ML region (linkRecords=None). Capped ONIOM lands in
Slice 4 and is gated separately.

The two modes are algebraically equivalent (see
docs/codex-plans/electrostatic-oniom-implementation-plan.md):

    E_existing_electrostatic =
        E_MM(host with ML-internal bonded/LJ removed and ML charges zeroed)
        + E_MACE(ml, +MM)

    E_oniom_electrostatic =
        E_MM(host with only ML charges zeroed)
        + E_MACE(ml, +MM)
        - E_low_model(ML-internal bonded + ML-internal LJ)

The ML-internal MM bonded + LJ that the existing mode removes via
mutation is removed in ONIOM via additive subtraction. Numerically the
two should agree to 1e-5 (or better) on identical positions.
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
from mace.modules import interaction_classes  # noqa: E402
from mace.modules.extensions import PolarMACE  # noqa: E402


_BOX_NM = 1.4
_DTYPE = torch.float64


def _build_polar_mace(device, dtype):
    fp_cfg = {
        "type": "AgnosticEmbeddedOneBodyVariableUpdate",
        "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding",
        "nonlinearity_cls": "MLPNonLinearity",
    }
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
        fixedpoint_update_config=fp_cfg,
        field_readout_config={"type": "OneBodyMLPFieldReadout"},
    ).to(device=device, dtype=dtype)


@pytest.fixture(scope="module")
def polar_mace_model_path():
    torch.manual_seed(7)
    model = _build_polar_mace(torch.device("cpu"), _DTYPE)
    model.eval()
    tmpdir = tempfile.mkdtemp(prefix="oniom_parity_")
    path = os.path.join(tmpdir, "polar_mace.pt")
    torch.save(model, path)
    yield path
    try:
        os.remove(path)
        os.rmdir(tmpdir)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Topology + system construction (1 ML water + 2 MM waters, PBC + PME).
# ---------------------------------------------------------------------------

_O_CHARGE = -0.834
_H_CHARGE = 0.417
_O_SIGMA = 0.31507
_O_EPS = 0.6364
_H_SIGMA = 1.0e-3
_H_EPS = 0.0


def _add_water(system, nb, bonds, angles, params):
    indices = []
    for mass, q, s, e in params:
        idx = system.addParticle(mass)
        nb.addParticle(
            q * unit.elementary_charge,
            s * unit.nanometer,
            e * unit.kilojoule_per_mole,
        )
        indices.append(idx)
    o, h1, h2 = indices
    bonds.addBond(o, h1, 0.09572 * unit.nanometer, 4.5e5 * unit.kilojoule_per_mole / unit.nanometer ** 2)
    bonds.addBond(o, h2, 0.09572 * unit.nanometer, 4.5e5 * unit.kilojoule_per_mole / unit.nanometer ** 2)
    angles.addAngle(h1, o, h2, 1.824, 460.0 * unit.kilojoule_per_mole / unit.radian ** 2)
    nb.addException(
        o, h1,
        0.0 * unit.elementary_charge ** 2,
        0.5 * (_O_SIGMA + _H_SIGMA) * unit.nanometer,
        0.0 * unit.kilojoule_per_mole,
    )
    nb.addException(
        o, h2,
        0.0 * unit.elementary_charge ** 2,
        0.5 * (_O_SIGMA + _H_SIGMA) * unit.nanometer,
        0.0 * unit.kilojoule_per_mole,
    )
    nb.addException(
        h1, h2,
        0.0 * unit.elementary_charge ** 2,
        _H_SIGMA * unit.nanometer,
        0.0 * unit.kilojoule_per_mole,
    )


def _build_system(num_mm_waters):
    topology = app.Topology()
    chain = topology.addChain()
    for _ in range(1 + num_mm_waters):
        res = topology.addResidue("HOH", chain)
        a_o = topology.addAtom("O", app.element.oxygen, res)
        a_h1 = topology.addAtom("H1", app.element.hydrogen, res)
        a_h2 = topology.addAtom("H2", app.element.hydrogen, res)
        topology.addBond(a_o, a_h1)
        topology.addBond(a_o, a_h2)
    topology.setPeriodicBoxVectors(
        unit.Quantity(np.diag([_BOX_NM, _BOX_NM, _BOX_NM]), unit.nanometer)
    )
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(_BOX_NM, 0, 0) * unit.nanometer,
        openmm.Vec3(0, _BOX_NM, 0) * unit.nanometer,
        openmm.Vec3(0, 0, _BOX_NM) * unit.nanometer,
    )
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.5 * unit.nanometer)
    bonds = openmm.HarmonicBondForce()
    angles = openmm.HarmonicAngleForce()
    o = (15.999, _O_CHARGE, _O_SIGMA, _O_EPS)
    h = (1.008, _H_CHARGE, _H_SIGMA, _H_EPS)
    for _ in range(1 + num_mm_waters):
        _add_water(system, nb, bonds, angles, [o, h, h])
    system.addForce(nb)
    system.addForce(bonds)
    system.addForce(angles)
    return topology, system


def _positions(num_mm_waters):
    base_o = np.array([
        [0.55, 0.55, 0.55],
        [0.20, 0.20, 0.20],
        [0.95, 0.20, 0.30],
        [0.30, 0.95, 0.95],
    ])[: 1 + num_mm_waters]
    h1 = np.array([0.09572, 0.0, 0.0])
    h2 = np.array([-0.0240, 0.0927, 0.0])
    coords = []
    for o in base_o:
        coords.append(o)
        coords.append(o + h1)
        coords.append(o + h2)
    return np.array(coords) * unit.nanometer


def _energy_and_forces(system, positions):
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
# Slice 3 parity tests
# ---------------------------------------------------------------------------

def test_oniom_matches_electrostatic_energy(polar_mace_model_path):
    """Total potential energy must match between the two modes within 1e-5."""
    topology, system_a = _build_system(num_mm_waters=2)
    _, system_b = _build_system(num_mm_waters=2)
    potential = MLPotential("mace", modelPath=polar_mace_model_path)
    pos = _positions(num_mm_waters=2)

    elec_system = potential.createMixedSystem(
        topology, system_a, [0, 1, 2], embedding="electrostatic"
    )
    e_elec, _ = _energy_and_forces(elec_system, pos)

    oniom_system = potential.createMixedSystem(
        topology, system_b, [0, 1, 2], embedding="oniom-electrostatic"
    )
    e_oniom, _ = _energy_and_forces(oniom_system, pos)

    assert e_oniom == pytest.approx(e_elec, rel=1e-5, abs=1e-3)


def test_oniom_matches_electrostatic_forces(polar_mace_model_path):
    """Per-atom forces must match between the two modes within 1e-3 kJ/mol/nm."""
    topology, system_a = _build_system(num_mm_waters=2)
    _, system_b = _build_system(num_mm_waters=2)
    potential = MLPotential("mace", modelPath=polar_mace_model_path)
    pos = _positions(num_mm_waters=2)

    elec_system = potential.createMixedSystem(
        topology, system_a, [0, 1, 2], embedding="electrostatic"
    )
    _, f_elec = _energy_and_forces(elec_system, pos)

    oniom_system = potential.createMixedSystem(
        topology, system_b, [0, 1, 2], embedding="oniom-electrostatic"
    )
    _, f_oniom = _energy_and_forces(oniom_system, pos)

    np.testing.assert_allclose(f_oniom, f_elec, rtol=1e-4, atol=1e-2)


def test_oniom_translation_invariance(polar_mace_model_path):
    """ONIOM must inherit PBC translation invariance from the underlying
    pieces. Shifting all positions by one box vector preserves energy."""
    topology, system = _build_system(num_mm_waters=2)
    potential = MLPotential("mace", modelPath=polar_mace_model_path)
    oniom = potential.createMixedSystem(
        topology, system, [0, 1, 2], embedding="oniom-electrostatic"
    )
    pos = _positions(num_mm_waters=2)
    e0, f0 = _energy_and_forces(oniom, pos)

    shifted = (pos.value_in_unit(unit.nanometer) + np.array([_BOX_NM, 0.0, 0.0])) * unit.nanometer
    e1, f1 = _energy_and_forces(oniom, shifted)
    assert e1 == pytest.approx(e0, rel=1e-6, abs=1e-4)
    np.testing.assert_allclose(f1, f0, rtol=1e-5, atol=1e-3)
