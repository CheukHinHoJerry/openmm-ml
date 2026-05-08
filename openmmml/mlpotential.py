"""
mlpotential.py: Provides a common API for creating OpenMM Systems with ML potentials.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2021-2026 Stanford University and the Authors.
Authors: Peter Eastman
Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import openmm
import openmm.app
import openmm.unit as unit
from copy import deepcopy
from typing import Dict, Iterable, Optional
import os
import shutil
import tempfile
import urllib.request
import sys
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


class MLPotentialImplFactory(object):
    """Abstract interface for classes that create MLPotentialImpl objects.

    If you are defining a new potential function, you need to create subclasses
    of MLPotentialImpl and MLPotentialImplFactory, and register an instance of
    the factory by calling MLPotential.registerImplFactory().  Alternatively,
    if a Python package creates an entry point in the group "openmmml.potentials",
    the potential will be registered automatically.  The entry point name is the
    name of the potential function, and the value should be the name of the
    MLPotentialImplFactory subclass.
    """
    
    def createImpl(self, name: str, **args) -> "MLPotentialImpl":
        """Create a MLPotentialImpl that will be used to implement a MLPotential.

        When a MLPotential is created, it invokes this method to create an object
        implementing the requested potential.  Subclasses must implement this method
        to return an instance of the correct MLPotentialImpl subclass.

        Parameters
        ----------
        name: str
            the name of the potential that was specified to the MLPotential constructor
        args:
            any additional keyword arguments that were provided to the MLPotential
            constructor are passed to this method.  This allows subclasses to customize
            their behavior based on extra arguments.

        Returns
        -------
        a MLPotentialImpl that implements the potential
        """
        raise NotImplementedError('Subclasses must implement createImpl()')


class MLPotentialImpl(object):
    """Abstract interface for classes that implement potential functions.

    If you are defining a new potential function, you need to create subclasses
    of MLPotentialImpl and MLPotentialImplFactory.  When a user creates a
    MLPotential and specifies a name for the potential to use, it looks up the
    factory that has been registered for that name and uses it to create a
    MLPotentialImpl of the appropriate subclass.
    """
    
    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  **args):
        """Add Force objects to a System to implement the potential function.

        This is invoked by MLPotential.createSystem().  Subclasses must implement
        it to create the requested potential function.

        Parameters
        ----------
        topology: Topology
            the Topology from which the System is being created
        system: System
            the System that is being created
        atoms: Optional[Iterable[int]]
            the indices of atoms the potential should be applied to, or None if
            it should be applied to the entire System
        forceGroup: int
            the force group that any newly added Forces should be in
        args:
            any additional keyword arguments that were provided to createSystem()
            are passed to this method.  This allows subclasses to customize their
            behavior based on extra arguments.
        """
        raise NotImplementedError('Subclasses must implement addForces()')

    def _getTorchDevice(self, args):
        """This is a utility routine for use by subclasses that are implemented with PyTorch.  It selects what device
        to use, which can either by specified by the user with the 'device' argument, or chosen automatically based on
        the available hardware."""
        import torch
        if 'device' in args:
            device = args['device']
            if isinstance(device, str):
                device = torch.device(device)
            return device
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _getCacheDir(self):
        """This is a utility routine returning a cache directory for use by subclasses that need to download their own
        models.  Before returning, it will create the directory and any parent directories if they do not already exist."""
        path = os.path.expanduser("~/.cache/openmm-ml")
        os.makedirs(path, exist_ok=True)
        return path

    def _downloadOrFindFile(self, name: str, url: str):
        """Downloads a file at the requested URL and saves it in a cache directory under the specified name.  If a file
        with the specified name already exists in the cache, the path to it will be returned without downloading it again."""
        cacheDir = self._getCacheDir()
        targetPath = os.path.join(cacheDir, name)
        if not os.path.isfile(targetPath):
            with urllib.request.urlopen(url) as remoteFile:
                # Download into a temporary directory first so that we do not
                # end up with a partially downloaded file at targetPath and we
                # do not create temporary files directly in the cache directory.
                with tempfile.TemporaryDirectory(dir=cacheDir) as tempDir:
                    tempPath = os.path.join(tempDir, name)
                    with open(tempPath, "wb") as localFile:
                        shutil.copyfileobj(remoteFile, localFile)
                    os.replace(tempPath, targetPath)
        return targetPath

class MLPotential(object):
    """A potential function that can be used in simulations.

    To use this class, create a MLPotential, specifying the name of the potential
    function to use.  You can then call createSystem() to create a System object
    for a simulation.  For example,

    >>> potential = MLPotential('ani2x')
    >>> system = potential.createSystem(topology)

    Alternatively, you can use createMixedSystem() to create a System where part is
    modeled with this potential and the rest is modeled with a conventional force
    field.  As an example, suppose the Topology contains three chains.  Chain 0 is
    a protein, chain 1 is a ligand, and chain 2 is solvent.  The following code
    creates a System in which the internal energy of the ligand is computed with
    ANI2x, while everything else (including interactions between the ligand and the
    rest of the System) is computed with Amber14.

    >>> forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    >>> mm_system = forcefield.createSystem(topology)
    >>> chains = list(topology.chains())
    >>> ml_atoms = [atom.index for atom in chains[1].atoms()]
    >>> potential = MLPotential('ani2x')
    >>> ml_system = potential.createMixedSystem(topology, mm_system, ml_atoms)
    """

    _implFactories: Dict[str, MLPotentialImplFactory] = {}
    
    def __init__(self, name: str, **args):
        """Create a MLPotential.

        Parameters
        ----------
        name: str
            the name of the potential function to use.  Built in support is currently
            provided for the following: 'ani1ccx', 'ani2x'.  Others may be added by
            calling MLPotential.registerImplFactory().
        args:
            particular potential functions may define additional arguments that can
            be used to customize them.  See the documentation on the specific
            potential functions for more information.
        """
        self._defaultArgs = dict(args)
        self._impl = MLPotential._implFactories[name].createImpl(name, **args)

    def createSystem(self, topology: openmm.app.Topology, removeCMMotion: bool = True, **args) -> openmm.System:
        """Create a System for running a simulation with this potential function.

        Parameters
        ----------
        topology: Topology
            the Topology for which to create a System
        removeCMMotion: bool
            if true, a CMMotionRemover will be added to the System. 
        args:
            particular potential functions may define additional arguments that can
            be used to customize them.  See the documentation on the specific
            potential functions for more information.

        Returns
        -------
        a newly created System object that uses this potential function to model the Topology
        """
        system = openmm.System()
        if topology.getPeriodicBoxVectors() is not None:
            system.setDefaultPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
        for atom in topology.atoms():
            if atom.element is None:
                system.addParticle(0)
            else:
                system.addParticle(atom.element.mass)
        mergedArgs = dict(self._defaultArgs)
        mergedArgs.update(args)
        self._impl.addForces(topology, system, None, 0, **mergedArgs)
        if removeCMMotion:
            system.addForce(openmm.CMMotionRemover())
        return system

    def createMixedSystem(self,
                          topology: openmm.app.Topology,
                          system: openmm.System,
                          atoms: Iterable[int],
                          removeConstraints: bool = True,
                          forceGroup: int = 0,
                          interpolate: bool = False,
                          **args) -> openmm.System:
        """Create a System that is partly modeled with this potential and partly
        with a conventional force field.

        To use this method, first create a System that is entirely modeled with the
        conventional force field.  Pass it to this method, along with the indices of the
        atoms to model with this potential (the "ML subset").  It returns a new System
        that is identical to the original one except for the following changes.

        1. Removing all bonds, angles, and torsions for which *all* atoms are in the
           ML subset.
        2. For every NonbondedForce and CustomNonbondedForce, adding exceptions/exclusions
           to prevent atoms in the ML subset from interacting with each other.
        3. (Optional) Removing constraints between atoms that are both in the ML subset.
        4. Adding Forces as necessary to compute the internal energy of the ML subset
           with this potential.

        Alternatively, the System can include Forces to compute the energy both with the
        conventional force field and with this potential, and to smoothly interpolate
        between them.  In that case, it creates a CustomCVForce containing the following.

        1. The Forces to compute this potential.
        2. Forces to compute the bonds, angles, and torsions that were removed above.
        3. For every NonbondedForce, a corresponding CustomBondForce to compute the
           nonbonded interactions within the ML subset.

        The CustomCVForce defines a global parameter called "lambda_interpolate" that interpolates
        between the two potentials.  When lambda_interpolate=0, the energy is computed entirely with
        the conventional force field.  When lambda_interpolate=1, the energy is computed entirely with
        the ML potential.  You can set its value by calling setParameter() on the Context.

        Parameters
        ----------
        topology: Topology
            the Topology for which to create a System
        system: System
            a System that models the Topology with a conventional force field
        atoms: Iterable[int]
            the indices of all atoms whose interactions should be computed with
            this potential
        removeConstraints: bool
            if True, remove constraints between pairs of atoms whose interaction
            will be computed with this potential
        forceGroup: int
            the force group the ML potential's Forces should be placed in
        interpolate: bool
            if True, create a System that can smoothly interpolate between the conventional
            and ML potentials
        args:
            particular potential functions may define additional arguments that can
            be used to customize them.  See the documentation on the specific
            potential functions for more information.

        Returns
        -------
        a newly created System object that uses this potential function to model the Topology
        """
        mergedArgs = dict(self._defaultArgs)
        mergedArgs.update(args)
        embedding = mergedArgs.get('embedding')
        if embedding == 'oniom-electrostatic':
            # Reject interpolate=True up front so a future implementation
            # cannot silently fall through into a CustomCVForce path that
            # doesn't compose with the model-Context PythonForce design.
            # See docs/codex-plans/electrostatic-oniom-redesign.md.
            if interpolate:
                raise ValueError(
                    "interpolate=True is not supported with "
                    "embedding='oniom-electrostatic'. The ONIOM low-model "
                    "correction is added as a separate PythonForce, not "
                    "inside the interpolation CustomCVForce."
                )
            return self._build_oniom_mixed_system(
                topology=topology,
                system=system,
                atoms=atoms,
                forceGroup=forceGroup,
                removeConstraints=removeConstraints,
                mergedArgs=mergedArgs,
            )
        electrostatic_embedding = embedding == 'electrostatic'
        if electrostatic_embedding and interpolate:
            raise ValueError(
                "interpolate=True is not currently supported with embedding='electrostatic'. "
                "Electrostatic embedding removes classical ML-MM Coulomb outside the "
                "interpolation CustomCVForce, so lambda_interpolate=0 would not reproduce "
                "the conventional MM endpoint."
            )

        # Remove ML-internal bonded terms. In electrostatic embedding, ML/MM
        # boundary bonded terms remain classical by default; only the classical
        # electrostatics replaced by the ML/MM coupling are removed below.
        # Classical ML-MM Lennard-Jones stays in the MM force field.

        newSystem = self._removeBonds(system, atoms, True, removeConstraints)

        atomList = list(atoms)
        for force in newSystem.getForces():
            if isinstance(force, openmm.NonbondedForce):
                if electrostatic_embedding:
                    atomSet = set(atomList)
                    numParticles = force.getNumParticles()
                    for i in range(numParticles):
                        charge, sigma, epsilon = force.getParticleParameters(i)
                        if i in atomSet:
                            # Zero only the direct Coulomb part on ML atoms.
                            # Sigma/epsilon stay so ML-MM LJ survives.
                            force.setParticleParameters(i, 0*charge, sigma, epsilon)
                    existing = {}
                    for i in range(force.getNumExceptions()):
                        p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(i)
                        existing[(int(p1), int(p2))] = (chargeProd, sigma, epsilon)
                    for i in range(numParticles):
                        i_in_ml = i in atomSet
                        for j in range(i):
                            if not (i_in_ml or j in atomSet):
                                continue
                            key = (i, j)
                            rev = (j, i)
                            if key in existing:
                                _, sigma, epsilon = existing[key]
                            elif rev in existing:
                                _, sigma, epsilon = existing[rev]
                            else:
                                _, sigma1, epsilon1 = force.getParticleParameters(i)
                                _, sigma2, epsilon2 = force.getParticleParameters(j)
                                sigma = 0.5*(sigma1+sigma2)
                                epsilon = unit.sqrt(epsilon1*epsilon2)
                            if i_in_ml and j in atomSet:
                                epsilon = 0*epsilon
                            force.addException(i, j, 0, sigma, epsilon, True)
                else:
                    for i in range(len(atomList)):
                        for j in range(i):
                            force.addException(atomList[i], atomList[j], 0, 1, 0, True)
            elif isinstance(force, openmm.CustomNonbondedForce):
                existing = set(tuple(force.getExclusionParticles(i)) for i in range(force.getNumExclusions()))
                if electrostatic_embedding:
                    for i in range(len(atomList)):
                        a1 = atomList[i]
                        for j in range(i):
                            a2 = atomList[j]
                            if (a1, a2) not in existing and (a2, a1) not in existing:
                                force.addExclusion(a1, a2)
                else:
                    for i in range(len(atomList)):
                        a1 = atomList[i]
                        for j in range(i):
                            a2 = atomList[j]
                            if (a1, a2) not in existing and (a2, a1) not in existing:
                                force.addExclusion(a1, a2)

        # Add the ML potential.

        if not interpolate:
            self._impl.addForces(topology, newSystem, atomList, forceGroup, **mergedArgs)
        else:
            # Create a CustomCVForce and put the ML forces inside it.

            cv = openmm.CustomCVForce('')
            cv.addGlobalParameter('lambda_interpolate', 1)
            tempSystem = openmm.System()
            self._impl.addForces(topology, tempSystem, atomList, forceGroup, **mergedArgs)
            mlVarNames = []
            for i, force in enumerate(tempSystem.getForces()):
                name = f'mlForce{i+1}'
                cv.addCollectiveVariable(name, deepcopy(force))
                mlVarNames.append(name)

            # Create Forces for all the bonded interactions within the ML subset and add them to the CustomCVForce.

            bondedSystem = self._removeBonds(system, atoms, False, removeConstraints)
            bondedForces = []
            for force in bondedSystem.getForces():
                if hasattr(force, 'addBond') or hasattr(force, 'addAngle') or hasattr(force, 'addTorsion'):
                    bondedForces.append(force)
            mmVarNames = []
            for i, force in enumerate(bondedForces):
                name = f'mmForce{i+1}'
                cv.addCollectiveVariable(name, deepcopy(force))
                mmVarNames.append(name)

            # Create a CustomBondForce that computes all nonbonded interactions within the ML subset.

            for force in system.getForces():
                if isinstance(force, openmm.NonbondedForce):
                    internalNonbonded = openmm.CustomBondForce('138.935456*chargeProd/r + 4*epsilon*((sigma/r)^12-(sigma/r)^6)')
                    internalNonbonded.addPerBondParameter('chargeProd')
                    internalNonbonded.addPerBondParameter('sigma')
                    internalNonbonded.addPerBondParameter('epsilon')
                    numParticles = system.getNumParticles()
                    atomCharge = [0]*numParticles
                    atomSigma = [0]*numParticles
                    atomEpsilon = [0]*numParticles
                    for i in range(numParticles):
                        charge, sigma, epsilon = force.getParticleParameters(i)
                        atomCharge[i] = charge
                        atomSigma[i] = sigma
                        atomEpsilon[i] = epsilon
                    exceptions = {}
                    for i in range(force.getNumExceptions()):
                        p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(i)
                        exceptions[(p1, p2)] = (chargeProd, sigma, epsilon)
                    for p1 in atomList:
                        for p2 in atomList:
                            if p1 == p2:
                                break
                            if (p1, p2) in exceptions:
                                chargeProd, sigma, epsilon = exceptions[(p1, p2)]
                            elif (p2, p1) in exceptions:
                                chargeProd, sigma, epsilon = exceptions[(p2, p1)]
                            else:
                                chargeProd = atomCharge[p1]*atomCharge[p2]
                                sigma = 0.5*(atomSigma[p1]+atomSigma[p2])
                                epsilon = unit.sqrt(atomEpsilon[p1]*atomEpsilon[p2])
                            if chargeProd._value != 0 or epsilon._value != 0:
                                internalNonbonded.addBond(p1, p2, [chargeProd, sigma, epsilon])
                    if internalNonbonded.getNumBonds() > 0:
                        name = f'mmForce{len(mmVarNames)+1}'
                        cv.addCollectiveVariable(name, internalNonbonded)
                        mmVarNames.append(name)

            # Configure the CustomCVForce so lambda_interpolate interpolates between the conventional and ML potentials.

            mlSum = '+'.join(mlVarNames) if len(mlVarNames) > 0 else '0'
            mmSum = '+'.join(mmVarNames) if len(mmVarNames) > 0 else '0'
            cv.setEnergyFunction(f'lambda_interpolate*({mlSum}) + (1-lambda_interpolate)*({mmSum})')
            newSystem.addForce(cv)
        return newSystem

    # ------------------------------------------------------------------
    # ONIOM-EE: model-Context PythonForce implementation (Slice 2′)
    # See docs/codex-plans/electrostatic-oniom-redesign.md.
    # ------------------------------------------------------------------

    def _build_oniom_mixed_system(
        self,
        topology: "openmm.app.Topology",
        system: openmm.System,
        atoms: Iterable[int],
        forceGroup: int,
        removeConstraints: bool,
        mergedArgs: dict,
    ) -> openmm.System:
        """Assemble a System for embedding='oniom-electrostatic'.

        Supports both closed-valence (`linkRecords=None`) and capped
        ML regions. The host MM System is left untouched. A model
        `Context` PythonForce, instantiated lazily on first force
        evaluation, evaluates the MM force field on the (possibly
        capped) ML subsystem and contributes -E_MM(model) /
        -F_MM(model) to the host. MACE provides E_high(model) via the
        existing addForces path.

        Total: E_total = E_MM(real) + E_MACE - E_MM(model).
        """
        if atoms is None:
            raise ValueError(
                "embedding='oniom-electrostatic' requires an explicit "
                "ml-atoms list."
            )

        atomList = [int(i) for i in atoms]

        # Normalize cap support args. linkRecords matches the existing
        # MACE link-atom contract; capMMParams is keyed by (q, m) and
        # provides MM force-field parameters for each cap atom.
        link_records_arg = mergedArgs.get('linkRecords')
        cap_mm_params_arg = mergedArgs.get('capMMParams')
        cap_info = self._oniom_normalize_caps(
            link_records_arg, cap_mm_params_arg, system, atomList,
            topology, mergedArgs,
        )

        # Host: deepcopy of source MM, no surgery. Bonded terms stay.
        # NonbondedForce stays. ML particle charges stay.
        host = deepcopy(system)
        if removeConstraints:
            self._oniom_remove_internal_constraints(host, set(atomList))

        # Add MACE PythonForce (+1) to the host. linkRecords flow
        # through to MACEPotentialImpl.addForces unchanged.
        self._impl.addForces(topology, host, atomList, forceGroup, **mergedArgs)

        # Build model System for E_MM(model). With caps, K extra
        # particles are appended at the end; their MM parameters come
        # from cap_info.
        model_system = _build_oniom_model_system(
            system, atomList, cap_info=cap_info,
        )

        # Wrap the model-Context evaluator in a PythonForce. Contributes
        # -E_MM(model) and -F_MM(model). Lazy Context instantiation
        # inside the closure (Codex review finding #2).
        PythonForce = getattr(openmm, "PythonForce", None)
        if PythonForce is None:
            PythonForce = getattr(getattr(openmm, "openmm", None), "PythonForce", None)
        if PythonForce is None:
            raise RuntimeError(
                "PythonForce is not available in this OpenMM build; "
                "embedding='oniom-electrostatic' requires it."
            )
        closure = _make_oniom_low_model_closure(
            model_system,
            num_atoms=system.getNumParticles(),
            cap_info=cap_info,
        )
        correction = PythonForce(closure)
        correction.setForceGroup(forceGroup)
        # Tell OpenMM this PythonForce needs box vectors when the model
        # `System` is periodic, otherwise the state passed to our closure
        # won't include them.
        host_is_periodic = (
            (topology.getPeriodicBoxVectors() is not None)
            or system.usesPeriodicBoundaryConditions()
        )
        correction.setUsesPeriodicBoundaryConditions(host_is_periodic)
        host.addForce(correction)

        return host

    @staticmethod
    def _oniom_remove_internal_constraints(
        new_system: openmm.System, atomSet: set
    ) -> None:
        """Remove constraints whose two atoms are both in the ML set."""
        for idx in reversed(range(new_system.getNumConstraints())):
            p1, p2, _ = new_system.getConstraintParameters(idx)
            if int(p1) in atomSet and int(p2) in atomSet:
                new_system.removeConstraint(idx)

    @staticmethod
    def _oniom_normalize_caps(
        link_records_arg,
        cap_mm_params_arg,
        system: openmm.System,
        atomList,
        topology,
        mergedArgs: dict,
    ):
        """Normalize linkRecords + capMMParams into an internal cap_info dict.

        Returns ``None`` when there are no caps (closed-valence, the
        Slice 2′ path). Returns a dict with arrays / lists when caps
        are present.

        cap_info schema:

            {
              "K":              int,
              "q_global":       np.ndarray of shape (K,) int64,
              "m_global":       np.ndarray of shape (K,) int64,
              "target_dist":    np.ndarray of shape (K,) float64 (nm),
              "cap_charge":     np.ndarray of shape (K,) float64 (e),
              "cap_sigma":      np.ndarray of shape (K,) float64 (nm),
              "cap_epsilon":    np.ndarray of shape (K,) float64 (kJ/mol),
            }

        capMMParams: optional dict keyed by (q, m) tuples, value is
        (charge_e, sigma_nm, epsilon_kjmol). Missing entries fall back
        to a generic aliphatic-H default ``(0.0, 0.106, 0.0656)``.

        Currently delegates to `MACEPotentialImpl._prepareLinkRecords`
        for record validation (uniqueness, q ∈ atoms, m ∉ atoms,
        non-PBC requirement, etc.) so we share the same enforcement
        as MACE itself.
        """
        if link_records_arg is None:
            if cap_mm_params_arg is not None:
                raise ValueError(
                    "capMMParams was provided but linkRecords is None. "
                    "Each cap MM-params entry must correspond to a "
                    "link record."
                )
            return None

        # Reuse MACE's validator. Imports done lazily because openmmml
        # users without MACE installed should still be able to import
        # mlpotential.
        from openmmml.models.macepotential import _prepareLinkRecords
        link_info = _prepareLinkRecords(
            link_records_arg, atomList, topology, system,
        )
        if link_info is None:
            return None

        K = int(link_info["K"])
        q_global = link_info["q_global"]
        m_global = link_info["m_global"]
        target_dist_ang = link_info["target_dist"]
        # MACE's _prepareLinkRecords stores target_dist in Angstroms
        # (because MACE's _computeMACE works in Å). The model-Context
        # closure here works in nm (OpenMM convention), so convert.
        target_dist_nm = np.asarray(target_dist_ang, dtype=np.float64) * 0.1

        # Default aliphatic-H force field params for the cap.
        DEFAULT_CHARGE_E = 0.0
        DEFAULT_SIGMA_NM = 0.106
        DEFAULT_EPSILON_KJ = 0.0656

        cap_charge = np.full(K, DEFAULT_CHARGE_E, dtype=np.float64)
        cap_sigma = np.full(K, DEFAULT_SIGMA_NM, dtype=np.float64)
        cap_epsilon = np.full(K, DEFAULT_EPSILON_KJ, dtype=np.float64)

        if cap_mm_params_arg is not None:
            if not isinstance(cap_mm_params_arg, dict):
                raise TypeError(
                    "capMMParams must be a dict keyed by (q, m) "
                    "tuples; got "
                    f"{type(cap_mm_params_arg).__name__}."
                )
            for k in range(K):
                key = (int(q_global[k]), int(m_global[k]))
                if key in cap_mm_params_arg:
                    q_e, sig_nm, eps_kj = cap_mm_params_arg[key]
                    cap_charge[k] = float(q_e)
                    cap_sigma[k] = float(sig_nm)
                    cap_epsilon[k] = float(eps_kj)

        # PBC + non-zero cap charges is currently unsupported (CRITICAL
        # caveat in the redesign plan): the model `Context` PME would
        # see N+K charges while the host PME sees N, breaking the
        # reciprocal-space cancellation. With cap charges = 0 the
        # model PME's structure factor over caps is zero, so the
        # asymmetry vanishes.
        host_is_periodic = (
            (topology.getPeriodicBoxVectors() is not None)
            or system.usesPeriodicBoundaryConditions()
        )
        if host_is_periodic and np.any(cap_charge != 0.0):
            raise NotImplementedError(
                "Capped ONIOM under PBC currently requires cap charges "
                "to be zero (the default). Non-zero cap charges break "
                "the reciprocal-space cancellation between E_MM(real) "
                "(N particles) and E_MM(model) (N+K particles). See "
                "docs/codex-plans/electrostatic-oniom-redesign.md "
                "(Slice 3′ design item)."
            )

        return {
            "K": K,
            "q_global": np.asarray(q_global, dtype=np.int64),
            "m_global": np.asarray(m_global, dtype=np.int64),
            "target_dist": target_dist_nm,
            "cap_charge": cap_charge,
            "cap_sigma": cap_sigma,
            "cap_epsilon": cap_epsilon,
        }

    def _removeBonds(self, system: openmm.System, atoms: Iterable[int], removeInSet: bool, removeConstraints: bool) -> openmm.System:
        """Copy a System, removing all bonded interactions between atoms in (or not in) a particular set.

        Parameters
        ----------
        system: System
            the System to copy
        atoms: Iterable[int]
            a set of atom indices
        removeInSet: bool
            if True, any bonded term connecting atoms in the specified set is removed.  If False,
            any term that does *not* connect atoms in the specified set is removed
        removeConstraints: bool
            if True, remove constraints between pairs of atoms in the set

        Returns
        -------
        a newly created System object in which the specified bonded interactions have been removed
        """
        atomSet = set(atoms)

        # Create an XML representation of the System.

        import xml.etree.ElementTree as ET
        xml = openmm.XmlSerializer.serialize(system)
        root = ET.fromstring(xml)

        # This function decides whether a bonded interaction should be removed.

        def shouldRemove(termAtoms):
            return all(a in atomSet for a in termAtoms) == removeInSet

        # Remove bonds, angles, and torsions.

        for bonds in root.findall('./Forces/Force/Bonds'):
            for bond in bonds.findall('Bond'):
                bondAtoms = [int(bond.attrib[p]) for p in ('p1', 'p2')]
                if shouldRemove(bondAtoms):
                    bonds.remove(bond)
        for angles in root.findall('./Forces/Force/Angles'):
            for angle in angles.findall('Angle'):
                angleAtoms = [int(angle.attrib[p]) for p in ('p1', 'p2', 'p3')]
                if shouldRemove(angleAtoms):
                    angles.remove(angle)
        for torsions in root.findall('./Forces/Force/Torsions'):
            for torsion in torsions.findall('Torsion'):
                torsionLabels =  ('p1', 'p2', 'p3', 'p4') if 'p1' in torsion.attrib else ('a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4')
                torsionAtoms = [int(torsion.attrib[p]) for p in torsionLabels]
                if shouldRemove(torsionAtoms):
                    torsions.remove(torsion)

        # Optionally remove constraints.

        if removeConstraints:
            for constraints in root.findall('./Constraints'):
                for constraint in constraints.findall('Constraint'):
                    constraintAtoms = [int(constraint.attrib[p]) for p in ('p1', 'p2')]
                    if shouldRemove(constraintAtoms):
                        constraints.remove(constraint)

        # Create a new System from it.

        return openmm.XmlSerializer.deserialize(ET.tostring(root, encoding='unicode'))

    @staticmethod
    def registerImplFactory(name: str, factory: MLPotentialImplFactory):
        """Register a new potential function that can be used with MLPotential.

        Parameters
        ----------
        name: str
            the name of the potential function that will be passed to the MLPotential constructor
        factory: MLPotentialImplFactory
            a factory object that will be used to create MLPotentialImpl objects
        """
        MLPotential._implFactories[name] = factory


# ---------------------------------------------------------------------
# ONIOM-EE helpers (Slice 2′ — closed-valence model-Context PythonForce)
# ---------------------------------------------------------------------

# NonbondedForce methods that are periodic (host evaluates Coulomb with
# Ewald split). The opposite-sign-PME trick used in the model System
# applies uniformly to both periodic and non-periodic methods, so this
# constant is currently informational.
_ONIOM_PERIODIC_NB_METHODS = frozenset({
    openmm.NonbondedForce.CutoffPeriodic,
    openmm.NonbondedForce.Ewald,
    openmm.NonbondedForce.PME,
    openmm.NonbondedForce.LJPME,
})


def _oniom_resolve_unsupported_bonded_like():
    """Bonded-like Force classes the builder does not handle. We refuse
    to silently drop their ML-internal entries because the host MM
    `System` still evaluates them, and a missing low-model subtraction
    would break the ONIOM cancellation."""
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
    return tuple(
        cls for cls in (getattr(openmm, n, None) for n in names) if cls is not None
    )


_ONIOM_UNSUPPORTED_BONDED_LIKE = _oniom_resolve_unsupported_bonded_like()


def _oniom_resolve_unsupported_nonbonded_like():
    """Nonbonded-like Force classes the builder does not handle. The
    builder supports `NonbondedForce` and `CustomNonbondedForce`; any
    other nonbonded-like force present in the source is refused
    because its ML-related contribution would not be subtracted by
    the low-model and the ONIOM cancellation would silently break."""
    names = (
        "CustomGBForce",
        "GBSAOBCForce",
        "AmoebaMultipoleForce",
        "AmoebaVdwForce",
        "AmoebaWcaDispersionForce",
        "AmoebaGeneralizedKirkwoodForce",
        "DrudeForce",
        "HippoNonbondedForce",
        "ATMForce",
    )
    return tuple(
        cls for cls in (getattr(openmm, n, None) for n in names) if cls is not None
    )


_ONIOM_UNSUPPORTED_NONBONDED_LIKE = _oniom_resolve_unsupported_nonbonded_like()


def _build_oniom_model_system(
    source: openmm.System,
    ml_atoms: Iterable[int],
    cap_info: Optional[dict] = None,
) -> openmm.System:
    """Build a `System` representing E_MM(model) for ONIOM-EE.

    Construction:
      1. Clone the source `System` via `XmlSerializer` (preserves PME
         parameters, box vectors, masses, particles bit-exactly — pinned
         by Slice 0's `TestOniomSlice0Prereq.py`).
      2. Strip every Force from the clone. If ``cap_info`` is provided,
         also append ``K`` cap particles at the end of the clone (mass
         = 1.008 amu, indices N..N+K-1 in the model `System`).
      3. Re-add bonded forces (HarmonicBond / HarmonicAngle /
         PeriodicTorsion) with entries restricted to ML-internal atom
         tuples. Each copy inherits `usesPeriodicBoundaryConditions()`
         from the source. (Cap-related bonded entries are NOT
         synthesized from the MM force field — the user's source
         topology has no Q-cap parameters, so the bonded picture of
         the cap is left to MACE on the high side.)
      4. Re-add **every** ``NonbondedForce`` and ``CustomNonbondedForce``
         from the source as an opposite-sign pair wrapped in a single
         ``CustomCVForce`` whose energy is the sum of ``(a_i - b_i)``
         over all clones. ``b_i`` clones get the ML-side surgery that
         removes the ML-related contribution. With caps present, both
         ``a_i`` and ``b_i`` clones have K extra particles appended
         using ``cap_info``'s MM parameters; the surgery on ``b_i``
         also zeros cap charges so cap-* Coulomb survives in
         ``a_i - b_i`` (matching the ML-* surgery semantics).

    Raises
    ------
    NotImplementedError
        If the source contains any bonded-like force class the builder
        does not support (CMAPTorsion, RBTorsion, custom bond/angle/
        torsion forces, etc.) OR any nonbonded-like force class
        beyond ``NonbondedForce`` / ``CustomNonbondedForce``
        (e.g. ``CustomGBForce``, ``AmoebaMultipoleForce``,
        ``DrudeForce``).
    """
    ml_set = set(int(i) for i in ml_atoms)
    K = 0 if cap_info is None else int(cap_info["K"])

    # 1. Full XML clone preserves PME params, box, particles, masses.
    clone_xml = openmm.XmlSerializer.serialize(source)
    model = openmm.XmlSerializer.deserialize(clone_xml)

    # Collect every supported nonbonded-like source force in order.
    source_nb_forces = [
        f for f in source.getForces()
        if isinstance(f, (openmm.NonbondedForce, openmm.CustomNonbondedForce))
    ]

    # Validate unsupported bonded-like AND nonbonded-like classes BEFORE
    # stripping so the error message reflects what the user actually has.
    for src_force in source.getForces():
        if isinstance(src_force, _ONIOM_UNSUPPORTED_BONDED_LIKE):
            raise NotImplementedError(
                f"embedding='oniom-electrostatic' does not support "
                f"{type(src_force).__name__} in the source System. "
                "Strip it before constructing the mixed system, or "
                "extend the builder."
            )
        if isinstance(src_force, _ONIOM_UNSUPPORTED_NONBONDED_LIKE):
            raise NotImplementedError(
                f"embedding='oniom-electrostatic' does not support "
                f"{type(src_force).__name__} in the source System. "
                "The ONIOM low-model would silently miss its "
                "ML-related contribution, breaking the cancellation. "
                "Strip it before constructing the mixed system, or "
                "extend the builder."
            )

    # 2. Strip all forces from the clone.
    while model.getNumForces() > 0:
        model.removeForce(0)

    # 3. Re-add bonded forces restricted to ML.
    for src_force in source.getForces():
        if isinstance(src_force, openmm.HarmonicBondForce):
            new = openmm.HarmonicBondForce()
            new.setUsesPeriodicBoundaryConditions(
                src_force.usesPeriodicBoundaryConditions()
            )
            for i in range(src_force.getNumBonds()):
                p1, p2, length, k = src_force.getBondParameters(i)
                if int(p1) in ml_set and int(p2) in ml_set:
                    new.addBond(int(p1), int(p2), length, k)
            if new.getNumBonds() > 0:
                model.addForce(new)
        elif isinstance(src_force, openmm.HarmonicAngleForce):
            new = openmm.HarmonicAngleForce()
            new.setUsesPeriodicBoundaryConditions(
                src_force.usesPeriodicBoundaryConditions()
            )
            for i in range(src_force.getNumAngles()):
                p1, p2, p3, theta, k = src_force.getAngleParameters(i)
                if all(int(p) in ml_set for p in (p1, p2, p3)):
                    new.addAngle(int(p1), int(p2), int(p3), theta, k)
            if new.getNumAngles() > 0:
                model.addForce(new)
        elif isinstance(src_force, openmm.PeriodicTorsionForce):
            new = openmm.PeriodicTorsionForce()
            new.setUsesPeriodicBoundaryConditions(
                src_force.usesPeriodicBoundaryConditions()
            )
            for i in range(src_force.getNumTorsions()):
                p1, p2, p3, p4, periodicity, phase, k = src_force.getTorsionParameters(i)
                if all(int(p) in ml_set for p in (p1, p2, p3, p4)):
                    new.addTorsion(
                        int(p1), int(p2), int(p3), int(p4), periodicity, phase, k
                    )
            if new.getNumTorsions() > 0:
                model.addForce(new)
        # Other force types (NonbondedForce, CMMotionRemover, etc.)
        # are intentionally not copied — the ONIOM low-model only
        # accounts for ML-internal bonded + ML-* nonbonded.

    # 3b. Append cap particles, if any. Cap atoms get hydrogen mass
    # and are added to the model `System` *before* nonbonded clones
    # are built so the clones see N+K particles.
    if K > 0:
        cap_atom_indices = []
        for _ in range(K):
            cap_atom_indices.append(model.addParticle(1.008))
        # Cap atom indices are appended at the end: N..N+K-1.
        cap_info_local = dict(cap_info)
        cap_info_local["model_indices"] = np.asarray(cap_atom_indices, dtype=np.int64)
        # Treat cap atoms as ML for the surgery (cap-* contributions
        # need to be subtracted from the host the same way ML-* are).
        ml_set_with_caps = ml_set | set(int(i) for i in cap_atom_indices)
    else:
        ml_set_with_caps = ml_set
        cap_info_local = None

    # 4. Re-add every supported nonbonded-like source force as an
    # opposite-sign pair inside a single CustomCVForce. The energy is
    # the sum of (a_i - b_i) over all clones; each pair contributes
    # exactly the ML-related portion of that source force.
    if source_nb_forces:
        cv = openmm.CustomCVForce("")
        terms = []
        for idx, src_force in enumerate(source_nb_forces):
            nb_xml = openmm.XmlSerializer.serialize(src_force)
            nb_a = openmm.XmlSerializer.deserialize(nb_xml)
            nb_b = openmm.XmlSerializer.deserialize(nb_xml)
            # Append K cap particles to BOTH clones so they have the
            # same particle count as the model `System`. Cap atoms get
            # explicit MM force-field params from cap_info.
            if K > 0:
                if isinstance(src_force, openmm.NonbondedForce):
                    for k in range(K):
                        nb_a.addParticle(
                            float(cap_info_local["cap_charge"][k]) * unit.elementary_charge,
                            float(cap_info_local["cap_sigma"][k]) * unit.nanometer,
                            float(cap_info_local["cap_epsilon"][k]) * unit.kilojoule_per_mole,
                        )
                        nb_b.addParticle(
                            float(cap_info_local["cap_charge"][k]) * unit.elementary_charge,
                            float(cap_info_local["cap_sigma"][k]) * unit.nanometer,
                            float(cap_info_local["cap_epsilon"][k]) * unit.kilojoule_per_mole,
                        )
                elif isinstance(src_force, openmm.CustomNonbondedForce):
                    # CustomNonbondedForce doesn't have a uniform charge
                    # parameter list; cap atoms get default
                    # per-particle parameters (zeros). Custom force
                    # fields with non-trivial cap params would need a
                    # follow-up.
                    nb_a_per_particle = nb_a.getNumPerParticleParameters()
                    nb_b_per_particle = nb_b.getNumPerParticleParameters()
                    for _ in range(K):
                        nb_a.addParticle([0.0] * nb_a_per_particle)
                        nb_b.addParticle([0.0] * nb_b_per_particle)
            if isinstance(src_force, openmm.NonbondedForce):
                _oniom_apply_nonbonded_surgery(nb_b, ml_set_with_caps)
            elif isinstance(src_force, openmm.CustomNonbondedForce):
                _oniom_apply_custom_nonbonded_surgery(nb_b, ml_set_with_caps)
            else:
                # Should not reach here; caught by the validation loop above.
                raise NotImplementedError(
                    f"unexpected nonbonded force class {type(src_force).__name__}"
                )
            a_name = f"nb_a_{idx}"
            b_name = f"nb_b_{idx}"
            cv.addCollectiveVariable(a_name, nb_a)
            cv.addCollectiveVariable(b_name, nb_b)
            terms.append(f"({a_name} - {b_name})")
        cv.setEnergyFunction(" + ".join(terms))
        model.addForce(cv)

    return model


def _oniom_apply_nonbonded_surgery(
    force: openmm.NonbondedForce, ml_set: set
) -> None:
    """Apply the existing electrostatic-mode surgery to a NonbondedForce.

    Mirrors the surgery the existing `embedding="electrostatic"` mode
    applies to the host NonbondedForce, but here we apply it only to
    the model `System`'s `nb_b` clone. The `nb_a - nb_b` subtraction
    inside the model `System`'s `CustomCVForce` then yields exactly
    the ML-related Coulomb + ML-internal LJ contributions to subtract.

    Modifications:
      - ML particle charges → 0 (sigma/epsilon kept).
      - For every ML-* atom pair: add an exception with chargeProd=0
        and (for ML-ML) epsilon=0; sigma/epsilon for ML-MM exceptions
        come from existing source exceptions or LB rules.
    """
    num_particles = force.getNumParticles()
    for i in range(num_particles):
        if i in ml_set:
            charge, sigma, epsilon = force.getParticleParameters(i)
            force.setParticleParameters(i, 0 * charge, sigma, epsilon)

    # Index existing exceptions (so we don't synthesize over them).
    existing = {}
    for i in range(force.getNumExceptions()):
        p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(i)
        existing[(int(p1), int(p2))] = (chargeProd, sigma, epsilon)

    for i in range(num_particles):
        i_in_ml = i in ml_set
        for j in range(i):
            if not (i_in_ml or j in ml_set):
                continue
            key = (i, j)
            rev = (j, i)
            if key in existing:
                _, sigma, epsilon = existing[key]
            elif rev in existing:
                _, sigma, epsilon = existing[rev]
            else:
                _, sigma1, epsilon1 = force.getParticleParameters(i)
                _, sigma2, epsilon2 = force.getParticleParameters(j)
                sigma = 0.5 * (sigma1 + sigma2)
                epsilon = unit.sqrt(epsilon1 * epsilon2)
            if i_in_ml and j in ml_set:
                epsilon = 0 * epsilon
            force.addException(i, j, 0, sigma, epsilon, True)


def _oniom_apply_custom_nonbonded_surgery(
    force: openmm.CustomNonbondedForce, ml_set: set
) -> None:
    """Apply the existing electrostatic-mode surgery to a CustomNonbondedForce.

    Mirrors what the existing ``embedding="electrostatic"`` mode does
    to the host (mlpotential.py L370-378): add an exclusion for every
    ML-ML pair so the ML-internal contribution evaluates to zero in
    the ``nb_b`` clone. ML-MM pairs are left as-is (they stay in the
    host and are *not* subtracted by the low-model — analogous to the
    treatment of ML-MM LJ in ``NonbondedForce``).

    The ``nb_a - nb_b`` subtraction therefore yields exactly the
    ML-internal CustomNonbondedForce energy.
    """
    existing = set()
    for i in range(force.getNumExclusions()):
        p1, p2 = force.getExclusionParticles(i)
        existing.add(_oniom_ordered_pair(int(p1), int(p2)))

    ml_list = sorted(ml_set)
    for i_idx, i in enumerate(ml_list):
        for j in ml_list[:i_idx]:
            key = _oniom_ordered_pair(i, j)
            if key not in existing:
                force.addExclusion(i, j)


def _oniom_ordered_pair(i: int, j: int):
    return (i, j) if i < j else (j, i)


def _make_oniom_low_model_closure(
    model_system: openmm.System,
    num_atoms: int,
    cap_info: Optional[dict] = None,
):
    """Return a callable suitable for `openmm.PythonForce`.

    Closure-bound state holds a lazily-instantiated model `Context`.
    First call builds the `Context` on the Reference platform; later
    calls reuse the cached `Context`.

    Returns negated energy/forces so `addForce(PythonForce(closure))`
    contributes `-E_MM(model)` and `-F_MM(model)` to the host.

    With caps (``cap_info != None``), per-step:
      1. Read host positions (``num_atoms`` rows).
      2. Compute K cap positions via the shared
         ``compute_cap_positions`` helper (same formula MACE uses on
         its side, so the same chemical region is consistent).
      3. Set positions on the model `Context` for ``num_atoms + K``
         particles.
      4. After ``getState``, redistribute cap forces back onto Q,M
         using ``redistribute_cap_force`` (the same Jacobian MACE's
         ``_computeMACE`` uses).
      5. Return ``num_atoms`` rows of forces (no cap rows in host).

    Notes
    -----
    The model `Context` defaults to OpenMM's Reference platform. For
    Slice 2′ / Slice 3′ this is a correctness-only choice; performance
    work (matching the host platform) is deferred.
    """
    import numpy as np

    state = {"model_context": None}
    K = 0 if cap_info is None else int(cap_info["K"])

    def closure(host_state):
        if state["model_context"] is None:
            platform = openmm.Platform.getPlatformByName("Reference")
            integrator = openmm.VerletIntegrator(0.001)
            state["model_context"] = openmm.Context(
                model_system, integrator, platform
            )

        ctx = state["model_context"]

        if model_system.usesPeriodicBoundaryConditions():
            box = host_state.getPeriodicBoxVectors(asNumpy=True)
            ctx.setPeriodicBoxVectors(box[0], box[1], box[2])

        host_positions = host_state.getPositions(asNumpy=True)
        if K == 0:
            ctx.setPositions(host_positions)
        else:
            # Compute cap positions from current Q,M using the shared
            # helper (matches MACE's _computeMACE side).
            from openmmml.embedding._links import (
                compute_cap_positions,
                redistribute_cap_force,
            )
            host_pos_nm = host_positions.value_in_unit(unit.nanometer) \
                if hasattr(host_positions, "value_in_unit") \
                else np.asarray(host_positions)
            r_Q = host_pos_nm[cap_info["q_global"]]
            r_M = host_pos_nm[cap_info["m_global"]]
            cap_pos, cap_C_L = compute_cap_positions(
                r_Q, r_M, cap_info["target_dist"]
            )
            full_pos = np.concatenate(
                [np.asarray(host_pos_nm, dtype=np.float64), cap_pos], axis=0
            )
            ctx.setPositions(full_pos * unit.nanometer)

        ms = ctx.getState(getEnergy=True, getForces=True)
        e = ms.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        f_full = ms.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        f_full = np.asarray(f_full, dtype=np.float64)

        if K == 0:
            f = f_full
        else:
            # Split host atoms vs cap atoms; redistribute cap forces
            # onto Q,M using the same Jacobian as MACE's _computeMACE.
            f = np.zeros((num_atoms, 3), dtype=np.float64)
            f[:] = f_full[:num_atoms]
            f_cap = f_full[num_atoms:num_atoms + K]
            F_Q_add, F_M_add = redistribute_cap_force(
                f_cap, r_Q, r_M, cap_C_L
            )
            # q_global / m_global are guaranteed unique by
            # _prepareLinkRecords, so plain += is safe.
            f[cap_info["q_global"]] += F_Q_add
            f[cap_info["m_global"]] += F_M_add

        # Sign flip: -E_MM(model), -F_MM(model)
        return -float(e), -f

    return closure


# Register any potential functions defined by entry points.

for potential in entry_points(group='openmmml.potentials'):
    MLPotential.registerImplFactory(potential.name, potential.load()())
