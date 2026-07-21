"""
electrostaticembedding.py: Implements electrostatic embedding.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2026 Stanford University and the Authors.
Authors: Peter Eastman, Evan Pretti
Contributors: Cheuk Hin Ho

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

from openmmml.mlpotential import MLPotentialImpl, Embedding, EmbeddingFactory
from openmmml.embeddings import utilities
import openmm
import openmm.app


class ElectrostaticEmbeddingFactory(EmbeddingFactory):
    """This is the factory that creates ElectrostaticEmbedding objects."""

    def createEmbedding(self, name: str, **args) -> Embedding:

        # name should always be "electrostatic" for this plugin.

        return ElectrostaticEmbedding()


class ElectrostaticEmbedding(Embedding):
    """Electrostatic embedding.  The ML model, rather than the conventional
    force field, is responsible for the electrostatic interactions between the
    atoms within the ML subset and those outside of it: it is passed the
    positions and conventional force field charges of the atoms outside the ML
    subset, and returns forces on them alongside the forces on the ML subset.

    This is implemented as the "global charge zero" variant: the conventional
    force field charge of every atom in the ML subset is set to zero.  Every
    Coulomb term involving an ML atom is then zero by construction, including
    the reciprocal-space part of PME, without any per-pair exceptions being
    added.  Adding an exception for each ML-MM pair would instead be incorrect
    under periodic boundary conditions, since NonbondedForce evaluates
    exceptions using plain Cartesian distances rather than the minimum image
    convention.  Lennard-Jones interactions between the ML subset and the rest
    of the system are left to the conventional force field and continue to use
    the ordinary, periodicity-aware pair list.

    Interactions within the ML subset are excluded entirely, as the ML model
    computes them.  Bonded terms that cross the ML/MM boundary are retained.

    Only a potential that can accept the charges and positions of the atoms
    outside the ML subset can be used with this embedding method.  The
    potential is passed ``embedding="electrostatic"`` and is responsible for
    reporting an error if it cannot honor it.
    """

    def __init__(self):
        pass

    def createMixedSystem(self,
                          potential: MLPotentialImpl,
                          topology: openmm.app.Topology,
                          system: openmm.System,
                          atoms: list[int],
                          forceGroup: int,
                          interpolate: bool,
                          **args) -> openmm.System:

        if interpolate:
            # At lambda_interpolate=0 the conventional endpoint would be missing
            # the ML-MM Coulomb energy, which is removed from the MM force field
            # outside of the interpolating CustomCVForce and cannot be restored
            # from within it.
            raise ValueError("Electrostatic embedding does not support interpolation.")

        periodic = system.usesPeriodicBoundaryConditions()

        # Create the new system with the ML-ML interactions that the ML
        # potential computes removed.

        newSystem = utilities.removeBonds(system, atoms, True)
        atomSet = set(atoms)

        for force in newSystem.getForces():
            if isinstance(force, openmm.NonbondedForce):

                # Zero the charge of every ML atom, which removes all Coulomb
                # interactions involving the ML subset while leaving its
                # Lennard-Jones parameters, and the MM-MM interactions,
                # untouched.

                for atom in atoms:
                    charge, sigma, epsilon = force.getParticleParameters(atom)
                    force.setParticleParameters(atom, 0.0, sigma, epsilon)

                # setParticleParameters() does not update the charge products
                # that were precomputed for existing exceptions, so the 1-4
                # Coulomb terms crossing the ML/MM boundary have to be zeroed
                # separately.

                for index in range(force.getNumExceptions()):
                    p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(index)
                    if p1 in atomSet or p2 in atomSet:
                        force.setExceptionParameters(index, p1, p2, 0.0, sigma, epsilon)

                # Exclude the ML-ML interactions entirely.

                for i in range(len(atoms)):
                    for j in range(i):
                        force.addException(atoms[i], atoms[j], 0, 1, 0, True)

                force.setExceptionsUsePeriodicBoundaryConditions(periodic)

            elif isinstance(force, openmm.CustomNonbondedForce):
                utilities.addCustomNonbondedExclusions(force, atoms)

        # Add the ML potential, telling it that it is responsible for the
        # electrostatic interactions with the atoms outside the ML subset.

        potential.addForces(topology, newSystem, atoms, forceGroup, embedding="electrostatic", **args)

        return newSystem
