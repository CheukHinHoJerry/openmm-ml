"""Regenerate the PolarMACE electrostatic-embedding reference in this directory.

``embedding="electrostatic"`` is only exercised end to end by a PolarMACE model,
which is not part of the openmm-ml test stack: it needs mace-torch with the
PolarMACE graft, graph_longrange, and a PolarMACE checkpoint.  Rather than skip
the check entirely, this script records a single-point energy and force set from
a real PolarMACE QM/MM run into ``polarmace_electrostatic_reference.txt``, so
that refactoring the embedding can be checked against a fixed number.

The system is a deterministically generated ~50-molecule TIP3P water cluster
with one water as the ML subset, built through amber14/tip3pfb.xml — small
enough to run on CPU in seconds, and identical on every run because
Modeller.addSolvent fills a lattice.

Run it as::

    python test/repro/polarmace_electrostatic_reference.py --model <PolarMACE.model>

and diff the result against the committed reference.  Pass ``--check`` to
compare against the committed file instead of overwriting it.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # the reference is a CPU run

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "polarmace_electrostatic_reference.txt"

QM_ATOMS = [0, 1, 2]  # the first water molecule
CHARGE, MULTIPLICITY = 0, 1
BOX_NM = 1.2


def buildSystem(workDir):
    """Build the water cluster topology, positions, and conventional System."""
    import numpy as np
    from openmm import app, unit

    forceField = app.ForceField("amber14/tip3pfb.xml")
    topology = app.Topology()
    box = (BOX_NM, BOX_NM, BOX_NM) * unit.nanometer
    topology.setUnitCellDimensions(box)
    modeller = app.Modeller(topology, np.zeros((0, 3)) * unit.nanometer)
    modeller.addSolvent(forceField, model="tip3p", boxSize=box)

    # Drop the periodic box.  linkRecords are only supported for non-periodic
    # systems; a water cluster has no bonds crossing the ML/MM boundary, so the
    # link record list comes out empty, but the topology must still read as
    # non-periodic for that code path to be reachable.
    modeller.topology.setPeriodicBoxVectors(None)

    pdb = workDir / "system.pdb"
    with open(pdb, "w") as file:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, file)
    manifest = workDir / "forcefield.json"
    manifest.write_text(json.dumps(
        {"base_forcefield": ["amber14/tip3pfb.xml"], "extra_forcefield_files": []}))
    return pdb, manifest


def computeReference(modelPath, workDir):
    import numpy as np
    import openmm
    from openmm import unit
    import mlmm.extensions.polarmace  # noqa: F401  grafts PolarMACE onto stock mace
    from openmmml import MLPotential
    from mlmm.link_atoms import infer_link_records_from_topology
    from mlmm_cli.build_mlmm_from_indices import build_system_from_manifest

    pdb, manifest = buildSystem(workDir)

    topology, positions, mmSystem = build_system_from_manifest(str(pdb), str(manifest))
    linkRecords = infer_link_records_from_topology(topology, QM_ATOMS)

    potential = MLPotential("mace", modelPath=str(modelPath),
                            charge=CHARGE, multiplicity=MULTIPLICITY)
    mixedSystem = potential.createMixedSystem(
        topology, mmSystem, QM_ATOMS,
        embedding="electrostatic", linkRecords=linkRecords, linkChargeScheme="dz1")

    integrator = openmm.VerletIntegrator(0.001*unit.picoseconds)
    context = openmm.Context(mixedSystem, integrator,
                             openmm.Platform.getPlatformByName("CPU"))
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole/unit.nanometer)

    nonbonded = next(force for force in mixedSystem.getForces()
                     if isinstance(force, openmm.NonbondedForce))
    charges = [nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
               for i in range(nonbonded.getNumParticles())]

    return {
        "numParticles": mmSystem.getNumParticles(),
        "energy": energy,
        "forces": np.asarray(forces),
        "numExceptions": nonbonded.getNumExceptions(),
        "mlCharges": [charges[i] for i in QM_ATOMS],
        "totalCharge": float(sum(charges)),
        "forceClasses": sorted(type(f).__name__ for f in mixedSystem.getForces()),
    }


def describeEnvironment(modelPath):
    import importlib.metadata
    import openmm
    import torch

    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    def gitDescribe(path):
        try:
            return subprocess.run(["git", "-C", str(path), "describe", "--always", "--dirty"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return "unknown"

    digest = hashlib.md5(Path(modelPath).read_bytes()).hexdigest()
    return [
        f"model               {Path(modelPath).name}",
        f"model md5           {digest}",
        f"openmm-ml           {gitDescribe(HERE.parents[1])}",
        f"python              {sys.version.split()[0]}",
        f"platform            {platform.platform()}",
        f"openmm              {openmm.version.version}",
        f"torch               {torch.__version__}",
        f"mace-torch          {version('mace-torch')}",
        f"graph_longrange     {version('graph_longrange')}",
        f"polarmace-mlmm      {version('polarmace-mlmm')}",
        f"numpy               {version('numpy')}",
    ]


def formatReference(result, environment):
    lines = [
        "PolarMACE electrostatic-embedding single-point reference",
        "=" * 72,
        "",
        "Regenerate with test/repro/polarmace_electrostatic_reference.py.",
        "System: deterministic 1.2 nm TIP3P water cluster (non-periodic),",
        "ML subset = atoms 0,1,2 (one water), embedding='electrostatic',",
        "linkChargeScheme='dz1', CPU platform, model default precision.",
        "",
        "Environment",
        "-" * 72,
    ]
    lines += environment
    lines += [
        "",
        "System",
        "-" * 72,
        f"particles           {result['numParticles']}",
        f"ML atoms            {QM_ATOMS}",
        f"forces              {', '.join(result['forceClasses'])}",
        f"NonbondedForce exceptions  {result['numExceptions']}",
        f"ML particle charges {result['mlCharges']}  (zeroed by the embedding)",
        f"total charge        {result['totalCharge']:.12g}",
        "",
        "Potential energy (kJ/mol)",
        "-" * 72,
        f"{result['energy']:.9f}",
        "",
        "Forces (kJ/mol/nm), one atom per line: index fx fy fz",
        "-" * 72,
    ]
    for index, force in enumerate(result["forces"]):
        lines.append(f"{index:5d} {force[0]:22.9f} {force[1]:22.9f} {force[2]:22.9f}")
    return "\n".join(lines) + "\n"


def parseReference(text):
    import numpy as np

    lines = text.splitlines()
    energyIndex = lines.index("Potential energy (kJ/mol)") + 2
    energy = float(lines[energyIndex])
    forceIndex = next(i for i, line in enumerate(lines) if line.startswith("Forces (")) + 2
    forces = np.array([[float(value) for value in line.split()[1:]]
                       for line in lines[forceIndex:] if line.strip()])
    return energy, forces


def main():
    import numpy as np
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to a PolarMACE checkpoint")
    parser.add_argument("--check", action="store_true",
                        help="compare against the committed reference instead of rewriting it")
    parser.add_argument("--energy-tolerance", type=float, default=1e-6,
                        help="absolute energy tolerance in kJ/mol for --check")
    parser.add_argument("--force-tolerance", type=float, default=1e-2,
                        help="absolute force tolerance in kJ/mol/nm for --check")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as workDir:
        result = computeReference(args.model, Path(workDir))

    if args.check:
        energy, forces = parseReference(REFERENCE.read_text())
        energyError = abs(result["energy"] - energy)
        forceError = float(np.abs(result["forces"] - forces).max())
        print(f"energy {result['energy']:.9f} kJ/mol (reference {energy:.9f}, "
              f"difference {energyError:.3e})")
        print(f"max force difference {forceError:.3e} kJ/mol/nm")
        if energyError > args.energy_tolerance or forceError > args.force_tolerance:
            raise SystemExit("reference mismatch")
        print("matches the committed reference")
    else:
        REFERENCE.write_text(formatReference(result, describeEnvironment(args.model)))
        print(f"wrote {REFERENCE}")
        print(f"energy {result['energy']:.9f} kJ/mol")


if __name__ == "__main__":
    main()
