"""Low-model `Force` builder for ONIOM-EE electrostatic embedding.

Slice 2 of docs/codex-plans/electrostatic-oniom-implementation-plan.md.

The ONIOM 2-layer extrapolation is

    E_total = E_high(ml, +MM_charges) + E_low(real) - E_low(model)

where `E_low(model)` is the MM-level energy of just the ML region,
including the ML-MM Coulomb evaluated against the MM charges as a fixed
background. This module builds the `Force` objects that, when added to
a host system with coefficient -1, evaluate `E_low(model)`.

The builder is independent of MACE; it operates purely on a source
`openmm.System` and a list of ML atom indices. Slice 3 will compose
these forces into a full ONIOM stack inside `MLPotential.createMixedSystem`.

Conventions
-----------
- Coulomb conversion factor matches OpenMM (`138.935456 kJ/mol nm e^-2`).
- ML-internal LJ and Coulomb are evaluated in **direct space** (`1/r`
  with no Ewald split). MACE's ML-MM Coulomb is also direct-space, so
  the two cancel exactly when both are present.
- ML-MM Coulomb is evaluated as `q_ml * q_mm / r` summed over all ML-MM
  pairs that are NOT excluded by a NonbondedForce exception in the
  source system (so the subtraction matches what `E_low(real)` actually
  contains).
- Bonded forces (HarmonicBond, HarmonicAngle, PeriodicTorsion) are
  copied with terms restricted to ML-internal atom tuples.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import openmm
import openmm.unit as unit


COULOMB_KJ_NM = 138.935456


class OniomLowModelBuilder:
    """Build the MM-level low-model energy of an ML subset as `Force`s.

    Parameters
    ----------
    source_system : openmm.System
        The unmodified MM `System`. Reads bonded, NonbondedForce
        particle parameters, and exceptions from this system.
    ml_atoms : Sequence[int]
        Indices of the ML atoms inside `source_system`.

    Notes
    -----
    The builder reads from `source_system` but does not modify it.
    Returned `Force` objects are independent of the source.
    """

    def __init__(self, source_system: openmm.System, ml_atoms: Sequence[int]):
        self._source = source_system
        self._ml = [int(i) for i in ml_atoms]
        self._ml_set: Set[int] = set(self._ml)
        self._num_particles = source_system.getNumParticles()

    # ---- low-model components ------------------------------------------

    def internal_bonded_forces(self) -> List[openmm.Force]:
        """Return new `Force` objects for ML-internal bonded terms.

        For each `HarmonicBondForce` / `HarmonicAngleForce` /
        `PeriodicTorsionForce` in the source, a new force of the same
        type is created containing only entries whose atom indices are
        all in the ML set.
        """
        out: List[openmm.Force] = []
        for source_force in self._source.getForces():
            if isinstance(source_force, openmm.HarmonicBondForce):
                out.append(self._copy_harmonic_bonds(source_force))
            elif isinstance(source_force, openmm.HarmonicAngleForce):
                out.append(self._copy_harmonic_angles(source_force))
            elif isinstance(source_force, openmm.PeriodicTorsionForce):
                out.append(self._copy_periodic_torsions(source_force))
        # Filter out empty forces so we don't attach no-op Force objects.
        return [f for f in out if _force_has_terms(f)]

    def internal_nonbonded_force(self) -> Optional[openmm.CustomBondForce]:
        """Return a `CustomBondForce` evaluating ML-internal LJ + direct-space
        Coulomb, or None if there is no `NonbondedForce` in the source.

        Mirrors the pair construction used by `mlpotential.py`'s
        `interpolate=True` branch: for each ML-ML pair, sigma/epsilon/
        chargeProd come from a NonbondedForce exception when present,
        otherwise from the per-particle parameters via Lorentz-Berthelot.
        """
        nb = self._find_nonbonded()
        if nb is None:
            return None

        force = openmm.CustomBondForce(
            f"{COULOMB_KJ_NM}*chargeProd/r + 4*epsilon*((sigma/r)^12-(sigma/r)^6)"
        )
        force.addPerBondParameter("chargeProd")
        force.addPerBondParameter("sigma")
        force.addPerBondParameter("epsilon")
        force.setUsesPeriodicBoundaryConditions(False)

        atom_charge, atom_sigma, atom_epsilon = self._read_particle_params(nb)
        exceptions = self._read_exceptions(nb)

        for i_idx, i in enumerate(self._ml):
            for j in self._ml[i_idx + 1 :]:
                key = _ordered_pair(i, j)
                if key in exceptions:
                    chargeProd, sigma, epsilon = exceptions[key]
                else:
                    chargeProd = atom_charge[i] * atom_charge[j]
                    sigma = 0.5 * (atom_sigma[i] + atom_sigma[j])
                    epsilon = (atom_epsilon[i] * atom_epsilon[j]) ** 0.5
                if chargeProd == 0.0 and epsilon == 0.0:
                    continue
                force.addBond(i, j, [chargeProd, sigma, epsilon])
        if force.getNumBonds() == 0:
            return None
        return force

    def ml_mm_coulomb_background_force(
        self,
        periodic: bool = False,
        cutoff: Optional[unit.Quantity] = None,
    ) -> Optional[openmm.CustomNonbondedForce]:
        """Return a `CustomNonbondedForce` for ML-MM direct-space Coulomb.

        Parameters
        ----------
        periodic : bool
            If True, the force uses `CutoffPeriodic` and the box vectors
            of whatever host `System` it is added to.
        cutoff : unit.Quantity, optional
            Cutoff distance. Required when `periodic=True`. Ignored
            when `periodic=False` (a NoCutoff force is used).
        """
        nb = self._find_nonbonded()
        if nb is None:
            return None

        atom_charge, _, _ = self._read_particle_params(nb)
        exceptions = self._read_exceptions(nb)

        force = openmm.CustomNonbondedForce(f"{COULOMB_KJ_NM}*charge1*charge2/r")
        force.addPerParticleParameter("charge")
        for i in range(self._num_particles):
            force.addParticle([atom_charge[i]])
        if periodic:
            if cutoff is None:
                raise ValueError("periodic=True requires a cutoff distance.")
            force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
            force.setCutoffDistance(cutoff)
        else:
            force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)

        ml_atoms = set(self._ml)
        mm_atoms = [i for i in range(self._num_particles) if i not in ml_atoms]
        if not mm_atoms:
            return None
        force.addInteractionGroup(self._ml, mm_atoms)

        # Mirror NonbondedForce exceptions: any (ml, mm) pair listed as an
        # exception is excluded so that direct-space Coulomb in the host
        # MM system also excludes it -- the subtraction stays exact.
        for (p1, p2), (chargeProd, _, _) in exceptions.items():
            in_ml_1 = p1 in ml_atoms
            in_ml_2 = p2 in ml_atoms
            if in_ml_1 != in_ml_2:
                force.addExclusion(p1, p2)
        return force

    def build_all(
        self,
        periodic: bool = False,
        cutoff: Optional[unit.Quantity] = None,
    ) -> List[openmm.Force]:
        """Convenience: return [bonded..., internal_nonbonded, ml_mm_coulomb]
        with `None` entries dropped.
        """
        forces: List[openmm.Force] = list(self.internal_bonded_forces())
        internal = self.internal_nonbonded_force()
        if internal is not None:
            forces.append(internal)
        background = self.ml_mm_coulomb_background_force(periodic=periodic, cutoff=cutoff)
        if background is not None:
            forces.append(background)
        return forces

    # ---- internals -----------------------------------------------------

    def _find_nonbonded(self) -> Optional[openmm.NonbondedForce]:
        for f in self._source.getForces():
            if isinstance(f, openmm.NonbondedForce):
                return f
        return None

    def _read_particle_params(
        self, nb: openmm.NonbondedForce
    ) -> Tuple[List[float], List[float], List[float]]:
        charge, sigma, epsilon = [], [], []
        for i in range(nb.getNumParticles()):
            q, s, e = nb.getParticleParameters(i)
            charge.append(q.value_in_unit(unit.elementary_charge))
            sigma.append(s.value_in_unit(unit.nanometer))
            epsilon.append(e.value_in_unit(unit.kilojoule_per_mole))
        return charge, sigma, epsilon

    def _read_exceptions(self, nb: openmm.NonbondedForce):
        out = {}
        for i in range(nb.getNumExceptions()):
            p1, p2, chargeProd, sigma, epsilon = nb.getExceptionParameters(i)
            out[_ordered_pair(int(p1), int(p2))] = (
                chargeProd.value_in_unit(unit.elementary_charge * unit.elementary_charge),
                sigma.value_in_unit(unit.nanometer),
                epsilon.value_in_unit(unit.kilojoule_per_mole),
            )
        return out

    def _copy_harmonic_bonds(
        self, source: openmm.HarmonicBondForce
    ) -> openmm.HarmonicBondForce:
        new = openmm.HarmonicBondForce()
        for i in range(source.getNumBonds()):
            p1, p2, length, k = source.getBondParameters(i)
            if int(p1) in self._ml_set and int(p2) in self._ml_set:
                new.addBond(int(p1), int(p2), length, k)
        return new

    def _copy_harmonic_angles(
        self, source: openmm.HarmonicAngleForce
    ) -> openmm.HarmonicAngleForce:
        new = openmm.HarmonicAngleForce()
        for i in range(source.getNumAngles()):
            p1, p2, p3, theta, k = source.getAngleParameters(i)
            if all(int(p) in self._ml_set for p in (p1, p2, p3)):
                new.addAngle(int(p1), int(p2), int(p3), theta, k)
        return new

    def _copy_periodic_torsions(
        self, source: openmm.PeriodicTorsionForce
    ) -> openmm.PeriodicTorsionForce:
        new = openmm.PeriodicTorsionForce()
        for i in range(source.getNumTorsions()):
            p1, p2, p3, p4, periodicity, phase, k = source.getTorsionParameters(i)
            if all(int(p) in self._ml_set for p in (p1, p2, p3, p4)):
                new.addTorsion(int(p1), int(p2), int(p3), int(p4), periodicity, phase, k)
        return new


def _ordered_pair(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _force_has_terms(force: openmm.Force) -> bool:
    if isinstance(force, openmm.HarmonicBondForce):
        return force.getNumBonds() > 0
    if isinstance(force, openmm.HarmonicAngleForce):
        return force.getNumAngles() > 0
    if isinstance(force, openmm.PeriodicTorsionForce):
        return force.getNumTorsions() > 0
    return True
