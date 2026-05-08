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

        Closed-valence path (linkRecords=None) only. The host MM System
        is left untouched. A model `Context` PythonForce, instantiated
        lazily on first force evaluation, evaluates the MM force field
        on the ML subsystem and contributes -E_MM(model) / -F_MM(model)
        to the host. MACE provides E_high(model) via the existing
        addForces path.

        Total: E_total = E_MM(real) + E_MACE - E_MM(model).
        """
        if mergedArgs.get('linkRecords') is not None:
            raise NotImplementedError(
                "Capped (link-atom) ONIOM is reserved for Slice 3′. The "
                "current implementation only supports closed-valence ML "
                "regions (linkRecords=None)."
            )
        if atoms is None:
            raise ValueError(
                "embedding='oniom-electrostatic' requires an explicit "
                "ml-atoms list."
            )

        atomList = [int(i) for i in atoms]

        # Host: deepcopy of source MM, no surgery. Bonded terms stay.
        # NonbondedForce stays. ML particle charges stay.
        host = deepcopy(system)
        if removeConstraints:
            self._oniom_remove_internal_constraints(host, set(atomList))

        # Add MACE PythonForce (+1) to the host.
        self._impl.addForces(topology, host, atomList, forceGroup, **mergedArgs)

        # Build model System for E_MM(model).
        model_system = _build_oniom_model_system(system, atomList)

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
    source: openmm.System, ml_atoms: Iterable[int]
) -> openmm.System:
    """Build a `System` representing E_MM(model) for ONIOM-EE.

    Construction:
      1. Clone the source `System` via `XmlSerializer` (preserves PME
         parameters, box vectors, masses, particles bit-exactly — pinned
         by Slice 0's `TestOniomSlice0Prereq.py`).
      2. Strip every Force from the clone.
      3. Re-add bonded forces (HarmonicBond / HarmonicAngle /
         PeriodicTorsion) with entries restricted to ML-internal atom
         tuples. Each copy inherits `usesPeriodicBoundaryConditions()`
         from the source.
      4. Re-add **every** ``NonbondedForce`` and ``CustomNonbondedForce``
         from the source as an opposite-sign pair wrapped in a single
         ``CustomCVForce`` whose energy is the sum of ``(a_i - b_i)``
         over all clones. ``b_i`` clones get the ML-side surgery that
         removes the ML-related contribution; ``a_i - b_i`` therefore
         evaluates exactly the ML-related contribution of that source
         force in matching functional form. Multiple source nonbonded
         forces (alchemical / layered setups) are all handled.

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
            if isinstance(src_force, openmm.NonbondedForce):
                _oniom_apply_nonbonded_surgery(nb_b, ml_set)
            elif isinstance(src_force, openmm.CustomNonbondedForce):
                _oniom_apply_custom_nonbonded_surgery(nb_b, ml_set)
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
    model_system: openmm.System, num_atoms: int
):
    """Return a callable suitable for `openmm.PythonForce`.

    Closure-bound state holds a lazily-instantiated model `Context`.
    First call builds the `Context` on the Reference platform; later
    calls reuse the cached `Context`.

    Returns negated energy/forces so `addForce(PythonForce(closure))`
    contributes `-E_MM(model)` and `-F_MM(model)` to the host.

    Notes
    -----
    The model `Context` defaults to OpenMM's Reference platform. For
    Slice 2′ this is a correctness-only choice; performance work
    (matching the host platform) is deferred. The model `System` is
    the same size as the host so the per-step cost on Reference is
    one PME evaluation comparable to the host's MM evaluation.
    """
    import numpy as np

    state = {"model_context": None}

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

        ctx.setPositions(host_state.getPositions(asNumpy=True))
        ms = ctx.getState(getEnergy=True, getForces=True)
        e = ms.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        f = ms.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        # Sign flip: -E_MM(model), -F_MM(model)
        return -float(e), -np.asarray(f, dtype=np.float64)

    return closure


# Register any potential functions defined by entry points.

for potential in entry_points(group='openmmml.potentials'):
    MLPotential.registerImplFactory(potential.name, potential.load()())
