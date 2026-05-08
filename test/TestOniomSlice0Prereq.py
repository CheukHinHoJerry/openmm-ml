"""Slice 0 prerequisites for the ONIOM-EE redesign.

See docs/codex-plans/electrostatic-oniom-redesign.md (Slice 0 section).

These tests pin two load-bearing assumptions of the planned model-
``Context`` ``PythonForce`` architecture:

1. Two ``PythonForce`` instances coexist in one ``System`` and both
   contribute their energies/forces. Slice 2' adds a second
   ``PythonForce`` (the model-``Context`` correction) alongside MACE's
   existing one.

2. ``XmlSerializer.serialize`` / ``deserialize`` round-trips PME
   parameters (``Ewald_alpha``, grid dims) bit-exactly. Slice 2' clones
   the host ``NonbondedForce`` for the model ``System`` and relies on
   PME parameters surviving the clone so the reciprocal sums agree.

Run these on ``mlmm`` before Slice 2' starts. If either fails, the
architecture needs a rethink.
"""
from __future__ import annotations

import math

import numpy as np
import openmm
import openmm.unit as unit
import pytest


# ---------------------------------------------------------------------------
# (1) Two PythonForces in one System
# ---------------------------------------------------------------------------

def _resolve_python_force_class():
    cls = getattr(openmm, "PythonForce", None)
    if cls is None:
        cls = getattr(getattr(openmm, "openmm", None), "PythonForce", None)
    return cls


def test_python_force_class_is_available():
    """OpenMM's PythonForce must be importable; both Slice 1's MACE
    addForces path and Slice 2''s correction depend on it."""
    cls = _resolve_python_force_class()
    assert cls is not None, (
        "openmm.PythonForce is not available in this OpenMM build. "
        "The ONIOM redesign architecture requires PythonForce; rebuild "
        "openmm with python-force support."
    )


def test_two_python_forces_in_one_system_both_contribute():
    """Add two PythonForces to one System and confirm both fire and
    their contributions sum into the total potential energy.

    Slice 2' adds a second PythonForce (model-Context correction)
    alongside MACE's existing one; this test is the smoke check that
    OpenMM does not silently ignore the second.
    """
    PythonForce = _resolve_python_force_class()
    if PythonForce is None:
        pytest.skip("openmm.PythonForce not available")

    system = openmm.System()
    n_atoms = 3
    for _ in range(n_atoms):
        system.addParticle(1.0)

    # Two closures with distinct, easily distinguishable contributions.
    # Each returns (energy_kJ_per_mol, forces_kJ_per_mol_per_nm shaped (N,3)).
    def force_a(state):
        forces = np.zeros((n_atoms, 3), dtype=np.float64)
        forces[0, 0] = 1.0
        return 7.5, forces

    def force_b(state):
        forces = np.zeros((n_atoms, 3), dtype=np.float64)
        forces[1, 1] = 2.0
        return 11.25, forces

    pf_a = PythonForce(force_a)
    pf_b = PythonForce(force_b)
    pf_a.setForceGroup(1)
    pf_b.setForceGroup(2)
    system.addForce(pf_a)
    system.addForce(pf_b)

    platform = openmm.Platform.getPlatformByName("Reference")
    integrator = openmm.VerletIntegrator(0.001)
    ctx = openmm.Context(system, integrator, platform)
    ctx.setPositions(np.zeros((n_atoms, 3)) * unit.nanometer)

    state = ctx.getState(getEnergy=True, getForces=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )

    # Energy: both contributions present.
    assert e == pytest.approx(7.5 + 11.25, rel=1e-12, abs=1e-9)

    # Force: each PythonForce wrote to a distinct atom/component.
    assert f[0, 0] == pytest.approx(1.0, rel=1e-12, abs=1e-9)
    assert f[1, 1] == pytest.approx(2.0, rel=1e-12, abs=1e-9)
    # Untouched components remain zero.
    assert f[2, 2] == pytest.approx(0.0, abs=1e-12)
    assert f[0, 1] == pytest.approx(0.0, abs=1e-12)


def test_two_python_forces_isolated_via_force_groups():
    """Force groups must isolate contributions: querying group 1 alone
    sees only force_a's energy, not force_b's. Slice 2' may rely on
    force-group bookkeeping to report the ONIOM correction separately."""
    PythonForce = _resolve_python_force_class()
    if PythonForce is None:
        pytest.skip("openmm.PythonForce not available")

    system = openmm.System()
    for _ in range(2):
        system.addParticle(1.0)

    def force_a(state):
        return 3.0, np.zeros((2, 3))

    def force_b(state):
        return 5.0, np.zeros((2, 3))

    pf_a = PythonForce(force_a)
    pf_b = PythonForce(force_b)
    pf_a.setForceGroup(1)
    pf_b.setForceGroup(2)
    system.addForce(pf_a)
    system.addForce(pf_b)

    platform = openmm.Platform.getPlatformByName("Reference")
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
    ctx.setPositions(np.zeros((2, 3)) * unit.nanometer)

    e_all = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    e_group_a = ctx.getState(
        getEnergy=True, groups={1}
    ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    e_group_b = ctx.getState(
        getEnergy=True, groups={2}
    ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    assert e_all == pytest.approx(8.0, abs=1e-9)
    assert e_group_a == pytest.approx(3.0, abs=1e-9)
    assert e_group_b == pytest.approx(5.0, abs=1e-9)


# ---------------------------------------------------------------------------
# (2) PME parameter XmlSerializer round-trip
# ---------------------------------------------------------------------------

def _build_pme_system(set_explicit_pme_params=True):
    """Return a 4-atom periodic System with a NonbondedForce(PME) where
    Ewald_alpha and grid dims are set explicitly when requested.
    """
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0)
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(2.0, 0, 0) * unit.nanometer,
        openmm.Vec3(0, 2.0, 0) * unit.nanometer,
        openmm.Vec3(0, 0, 2.0) * unit.nanometer,
    )
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.6 * unit.nanometer)
    if set_explicit_pme_params:
        # Pick non-default values so the test would fail if the round-trip
        # silently reset to defaults.
        nb.setPMEParameters(0.7 / unit.nanometer, 32, 32, 32)
    for q in (0.5, -0.3, 0.1, -0.3):
        nb.addParticle(
            q * unit.elementary_charge,
            0.3 * unit.nanometer,
            0.2 * unit.kilojoule_per_mole,
        )
    system.addForce(nb)
    return system


def _read_pme_params(nb):
    """Return ``(alpha_per_nm, nx, ny, nz)`` as plain Python floats / ints."""
    alpha, nx, ny, nz = nb.getPMEParameters()
    alpha_value = alpha.value_in_unit(unit.nanometer ** -1)
    return float(alpha_value), int(nx), int(ny), int(nz)


def test_xml_serializer_round_trip_preserves_pme_parameters():
    """Slice 2' clones the host NonbondedForce via ``XmlSerializer`` (the
    same trick ``_removeBonds`` uses) to build the model ``System``. PME
    alpha and grid dims must survive the round-trip bit-exactly so the
    reciprocal sum matches the host.
    """
    src = _build_pme_system(set_explicit_pme_params=True)
    src_nb = next(
        f for f in src.getForces() if isinstance(f, openmm.NonbondedForce)
    )
    src_alpha, src_nx, src_ny, src_nz = _read_pme_params(src_nb)
    src_cutoff = src_nb.getCutoffDistance().value_in_unit(unit.nanometer)
    src_method = src_nb.getNonbondedMethod()

    xml = openmm.XmlSerializer.serialize(src)
    rt = openmm.XmlSerializer.deserialize(xml)
    rt_nb = next(
        f for f in rt.getForces() if isinstance(f, openmm.NonbondedForce)
    )
    rt_alpha, rt_nx, rt_ny, rt_nz = _read_pme_params(rt_nb)
    rt_cutoff = rt_nb.getCutoffDistance().value_in_unit(unit.nanometer)
    rt_method = rt_nb.getNonbondedMethod()

    assert rt_method == src_method
    assert rt_cutoff == pytest.approx(src_cutoff, rel=1e-15, abs=1e-15)
    assert rt_alpha == pytest.approx(src_alpha, rel=1e-15, abs=1e-15)
    assert (rt_nx, rt_ny, rt_nz) == (src_nx, src_ny, src_nz)


def test_xml_serializer_round_trip_preserves_default_pme_parameters():
    """Same round-trip on a NonbondedForce that does NOT call
    setPMEParameters — i.e., relies on OpenMM's defaults. Both source
    and round-trip should report the same params (whatever the default
    is), and the round-trip should not silently drift.
    """
    src = _build_pme_system(set_explicit_pme_params=False)
    src_nb = next(
        f for f in src.getForces() if isinstance(f, openmm.NonbondedForce)
    )

    xml = openmm.XmlSerializer.serialize(src)
    rt = openmm.XmlSerializer.deserialize(xml)
    rt_nb = next(
        f for f in rt.getForces() if isinstance(f, openmm.NonbondedForce)
    )
    assert _read_pme_params(rt_nb) == _read_pme_params(src_nb)


def test_xml_serializer_round_trip_pme_in_context_matches():
    """Bit-exact-ness inside an actual ``Context`` is what matters for
    Slice 2''s PME cancellation. Build a Context on the source and on
    the round-trip, query ``getPMEParametersInContext()``, and compare.
    """
    src = _build_pme_system(set_explicit_pme_params=True)
    xml = openmm.XmlSerializer.serialize(src)
    rt = openmm.XmlSerializer.deserialize(xml)

    src_nb = next(
        f for f in src.getForces() if isinstance(f, openmm.NonbondedForce)
    )
    rt_nb = next(
        f for f in rt.getForces() if isinstance(f, openmm.NonbondedForce)
    )

    platform = openmm.Platform.getPlatformByName("Reference")
    src_ctx = openmm.Context(src, openmm.VerletIntegrator(0.001), platform)
    rt_ctx = openmm.Context(rt, openmm.VerletIntegrator(0.001), platform)
    src_ctx.setPositions(np.zeros((4, 3)) * unit.nanometer)
    rt_ctx.setPositions(np.zeros((4, 3)) * unit.nanometer)

    src_alpha_in_ctx, src_nx, src_ny, src_nz = src_nb.getPMEParametersInContext(src_ctx)
    rt_alpha_in_ctx, rt_nx, rt_ny, rt_nz = rt_nb.getPMEParametersInContext(rt_ctx)

    # getPMEParametersInContext returns alpha as a plain float in 1/nm
    # (unlike getPMEParameters which returns a Quantity). Strip Quantity
    # wrapping defensively in case the API changes.
    src_alpha_value = float(getattr(src_alpha_in_ctx, "_value", src_alpha_in_ctx))
    rt_alpha_value = float(getattr(rt_alpha_in_ctx, "_value", rt_alpha_in_ctx))

    assert math.isclose(rt_alpha_value, src_alpha_value, rel_tol=1e-15, abs_tol=1e-15), (
        f"PME alpha differs in Context: src={src_alpha_value} rt={rt_alpha_value}"
    )
    assert (int(rt_nx), int(rt_ny), int(rt_nz)) == (
        int(src_nx),
        int(src_ny),
        int(src_nz),
    )
