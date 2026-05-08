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
- ML-internal LJ and the **non-PBC** Coulomb pieces are evaluated in
  direct space (`1/r`, no Ewald split). This matches the host MM
  ``NonbondedForce`` when ``NonbondedMethod == NoCutoff`` and matches
  MACE's ``_ml_mm_coulomb`` direct-pair-sum path.
- **PBC Coulomb is intentionally not supported by this builder.** Under
  PBC the host MM ``NonbondedForce`` uses PME (``erfc(αr)/r`` direct +
  reciprocal Ewald), and MACE uses a GTO k-space evaluator
  (``GTOElectrostaticEnergy`` from ``graph_longrange``). Neither
  matches a static ``CustomBondForce`` / ``CustomNonbondedForce``
  ``1/r`` form, so a Coulomb-bearing low-model under PBC would leave a
  non-physical residual in the ONIOM cancellation. Under PBC the
  Coulomb-bearing methods raise ``NotImplementedError`` pointing to
  Slice 5 (purely-additive PME via the opposite-sign-PME trick).
  ``internal_bonded_forces()`` remains valid under PBC because it
  involves no Ewald split.
- ML-MM Coulomb (non-PBC only) is evaluated as ``q_ml * q_mm / r``
  summed over all ML-MM pairs that are NOT excluded by a
  ``NonbondedForce`` exception in the source system (so the
  subtraction matches what ``E_low(real)`` actually contains).
- Bonded forces (HarmonicBond, HarmonicAngle, PeriodicTorsion) are
  copied with terms restricted to ML-internal atom tuples and inherit
  ``usesPeriodicBoundaryConditions()`` from the source.

See https://github.com/CheukHinHoJerry/openmm-ml/issues for tracking
the PBC limitation.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import openmm
import openmm.unit as unit


COULOMB_KJ_NM = 138.935456


_PERIODIC_NB_METHODS = frozenset(
    {
        openmm.NonbondedForce.CutoffPeriodic,
        openmm.NonbondedForce.Ewald,
        openmm.NonbondedForce.PME,
        openmm.NonbondedForce.LJPME,
    }
)


# Force classes that are "bonded-like" (their entries can refer to
# ML-internal atom tuples that the closed-valence low-model would have
# to subtract) but are not handled by `internal_bonded_forces`. The
# builder raises when a source system contains any of these so the
# caller is forced to either remove them or extend the builder.
#
# isinstance() is used (not class-name string comparison) so subclasses
# in plugin code are also caught. `getattr(openmm, name, None)` keeps
# the tuple resilient across OpenMM versions where some classes may
# not be present.
def _resolve_unsupported_bonded_like():
    names = (
        "CMAPTorsionForce",
        "RBTorsionForce",
        "CustomBondForce",
        "CustomAngleForce",
        "CustomTorsionForce",
        "CustomCompoundBondForce",
        "CustomCentroidBondForce",
        "AmoebaTorsionTorsionForce",
        "GayBerneForce",
    )
    return tuple(cls for cls in (getattr(openmm, n, None) for n in names) if cls is not None)


_UNSUPPORTED_BONDED_LIKE = _resolve_unsupported_bonded_like()


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
        all in the ML set. Each copy inherits
        `usesPeriodicBoundaryConditions()` from its source so bonded
        tuples spanning the unit-cell boundary use the same displacement
        as the host MM system.

        Raises
        ------
        NotImplementedError
            If the source contains a bonded-like force class the
            builder does not support (e.g. `CMAPTorsionForce`,
            `RBTorsionForce`, `CustomBondForce`). Silently dropping
            those would let the low-model omit a contribution that the
            host MM system still evaluates, which would break the
            ONIOM cancellation.
        """
        out: List[openmm.Force] = []
        for source_force in self._source.getForces():
            if isinstance(source_force, openmm.HarmonicBondForce):
                out.append(self._copy_harmonic_bonds(source_force))
            elif isinstance(source_force, openmm.HarmonicAngleForce):
                out.append(self._copy_harmonic_angles(source_force))
            elif isinstance(source_force, openmm.PeriodicTorsionForce):
                out.append(self._copy_periodic_torsions(source_force))
            elif isinstance(source_force, _UNSUPPORTED_BONDED_LIKE):
                raise NotImplementedError(
                    f"OniomLowModelBuilder does not support "
                    f"{type(source_force).__name__} in the source System. "
                    "Extend the builder before using ONIOM-EE on a system "
                    "with this force type, or strip it before constructing "
                    "the builder."
                )
        # Filter out empty forces so we don't attach no-op Force objects.
        return [f for f in out if _force_has_terms(f)]

    def internal_nonbonded_force(self) -> Optional[openmm.CustomBondForce]:
        """Return a `CustomBondForce` evaluating ML-internal LJ + direct-space
        Coulomb, or None if there is no `NonbondedForce` in the source.

        Mirrors the pair construction used by `mlpotential.py`'s
        `interpolate=True` branch: for each ML-ML pair, sigma/epsilon/
        chargeProd come from a NonbondedForce exception when present,
        otherwise from the per-particle parameters via Lorentz-Berthelot.

        Raises
        ------
        NotImplementedError
            If the source `NonbondedForce` uses a periodic method
            (PME / Ewald / CutoffPeriodic / LJPME). A direct-space `1/r`
            CustomBondForce cannot reproduce PME's Ewald split, and
            substituting it would leave a non-physical residual in the
            ONIOM cancellation. See module docstring and Slice 5.
        """
        nb = self._find_nonbonded()
        if nb is None:
            return None
        _raise_if_periodic_coulomb(nb, "internal_nonbonded_force")

        force = openmm.CustomBondForce(
            f"{COULOMB_KJ_NM}*chargeProd/r + 4*epsilon*((sigma/r)^12-(sigma/r)^6)"
        )
        force.addPerBondParameter("chargeProd")
        force.addPerBondParameter("sigma")
        force.addPerBondParameter("epsilon")
        # Non-PBC source by construction (the periodic case raised above);
        # CustomBondForce keeps its default usesPeriodicBoundaryConditions=False.

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
        periodic: Optional[bool] = None,
        cutoff: Optional[unit.Quantity] = None,
    ) -> Optional[openmm.CustomNonbondedForce]:
        """Return a `CustomNonbondedForce` for ML-MM direct-space Coulomb.

        Non-PBC only. The PBC path is gated by `NotImplementedError`;
        see the module docstring for the rationale (PME vs `1/r`
        functional-form mismatch).

        Parameters
        ----------
        periodic : bool, optional
            Must be `False` or `None` (auto-derived from source). A
            `True` value, or a periodic source, raises
            `NotImplementedError`.
        cutoff : unit.Quantity, optional
            Ignored under non-PBC (a NoCutoff force is produced).
        """
        nb = self._find_nonbonded()
        if nb is None:
            return None
        _raise_if_periodic_coulomb(nb, "ml_mm_coulomb_background_force")

        src_is_periodic = nb.getNonbondedMethod() in _PERIODIC_NB_METHODS
        if periodic is None:
            periodic = src_is_periodic
        if periodic:
            raise NotImplementedError(
                "ml_mm_coulomb_background_force(periodic=True) is not "
                "supported. Direct-space 1/r cannot reproduce the host "
                "MM PME / Ewald / LJPME split. See Slice 5 of the plan."
            )
        # cutoff is ignored under non-PBC: a NoCutoff CustomNonbondedForce
        # is emitted below.
        del cutoff

        atom_charge, _, _ = self._read_particle_params(nb)
        exceptions = self._read_exceptions(nb)

        force = openmm.CustomNonbondedForce(f"{COULOMB_KJ_NM}*charge1*charge2/r")
        force.addPerParticleParameter("charge")
        for i in range(self._num_particles):
            force.addParticle([atom_charge[i]])
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
        periodic: Optional[bool] = None,
        cutoff: Optional[unit.Quantity] = None,
    ) -> List[openmm.Force]:
        """Convenience: return [bonded..., internal_nonbonded, ml_mm_coulomb]
        with `None` entries dropped. When `periodic`/`cutoff` are `None`
        (default), they are auto-derived from the source `NonbondedForce`.
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
        # Inherit PBC: a bonded tuple that spans the unit-cell boundary uses
        # minimum-image displacement only when usesPeriodicBoundaryConditions
        # is True; if we leave it at the default (False) the copy disagrees
        # with the source on those tuples and the low-model subtraction
        # diverges from the actual MM bonded contribution.
        new.setUsesPeriodicBoundaryConditions(source.usesPeriodicBoundaryConditions())
        for i in range(source.getNumBonds()):
            p1, p2, length, k = source.getBondParameters(i)
            if int(p1) in self._ml_set and int(p2) in self._ml_set:
                new.addBond(int(p1), int(p2), length, k)
        return new

    def _copy_harmonic_angles(
        self, source: openmm.HarmonicAngleForce
    ) -> openmm.HarmonicAngleForce:
        new = openmm.HarmonicAngleForce()
        new.setUsesPeriodicBoundaryConditions(source.usesPeriodicBoundaryConditions())
        for i in range(source.getNumAngles()):
            p1, p2, p3, theta, k = source.getAngleParameters(i)
            if all(int(p) in self._ml_set for p in (p1, p2, p3)):
                new.addAngle(int(p1), int(p2), int(p3), theta, k)
        return new

    def _copy_periodic_torsions(
        self, source: openmm.PeriodicTorsionForce
    ) -> openmm.PeriodicTorsionForce:
        new = openmm.PeriodicTorsionForce()
        new.setUsesPeriodicBoundaryConditions(source.usesPeriodicBoundaryConditions())
        for i in range(source.getNumTorsions()):
            p1, p2, p3, p4, periodicity, phase, k = source.getTorsionParameters(i)
            if all(int(p) in self._ml_set for p in (p1, p2, p3, p4)):
                new.addTorsion(int(p1), int(p2), int(p3), int(p4), periodicity, phase, k)
        return new


def _raise_if_periodic_coulomb(
    nb: openmm.NonbondedForce, method_name: str
) -> None:
    """Gate the Coulomb-bearing low-model methods under PBC.

    Under PBC the host MM `NonbondedForce` uses PME / Ewald / LJPME,
    whose direct part is `erfc(αr)/r` and which carries a non-trivial
    reciprocal-space tail. A static `CustomBondForce` /
    `CustomNonbondedForce` `1/r` cannot reproduce that, so substituting
    it would leave a non-physical residual in the ONIOM cancellation.
    Slice 5 will handle PBC via the opposite-sign-PME trick.

    `internal_lj_force()` and `internal_bonded_forces()` involve no
    Ewald split and remain valid under PBC; only Coulomb-bearing
    methods are gated.
    """
    if nb.getNonbondedMethod() in _PERIODIC_NB_METHODS:
        raise NotImplementedError(
            f"{method_name}() does not support a periodic source "
            f"NonbondedForce (got method={nb.getNonbondedMethod()}). "
            "PME / Ewald / LJPME hosts cannot be cancelled by a "
            "direct-space 1/r low-model. Bonded copies remain valid "
            "under PBC; for the full ONIOM stack on a periodic host, "
            "wait for Slice 5 (purely-additive PME). See module "
            "docstring for details."
        )


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
