import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest
import torch

from openmmml import MLPotential
from openmmml.models.macepotential import (
    _computeMACE,
    _min_image,
    _prepareLinkRecords,
    _prepareMMEmbedding,
    _removeMLMMElectrostatics,
    _should_use_mm_embedding,
)

mace = pytest.importorskip("mace", reason="mace is not installed")
platform_ints = range(mm.Platform.getNumPlatforms())
# Get the path to the test data
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _simple_nonbonded_system():
    system = mm.System()
    nonbonded = mm.NonbondedForce()
    params = [
        (1.0, 0.30, 0.20),
        (-0.5, 0.40, 0.50),
        (0.25, 0.50, 0.80),
    ]
    for charge_e, sigma_nm, epsilon_kj in params:
        system.addParticle(12.0)
        nonbonded.addParticle(
            charge_e * unit.elementary_charge,
            sigma_nm * unit.nanometer,
            epsilon_kj * unit.kilojoule_per_mole,
        )
    system.addForce(nonbonded)
    return system, nonbonded


def testPrepareMMEmbeddingAndRemoveMLMMElectrostatics():
    system, nonbonded = _simple_nonbonded_system()
    info = _prepareMMEmbedding(system, [0, 1])
    assert info is not None
    np.testing.assert_array_equal(info["ml_atoms"], [0, 1])
    np.testing.assert_array_equal(info["mm_atoms"], [2])
    np.testing.assert_allclose(info["mm_charges"], [0.25], atol=1e-12)

    _removeMLMMElectrostatics(system, info)

    expected_sigma_02 = 0.5 * (0.30 + 0.50)
    expected_sigma_12 = 0.5 * (0.40 + 0.50)
    expected_eps_02 = np.sqrt(0.20 * 0.80)
    expected_eps_12 = np.sqrt(0.50 * 0.80)
    seen = {}
    for i in range(nonbonded.getNumExceptions()):
        p1, p2, chargeProd, sigma, epsilon = nonbonded.getExceptionParameters(i)
        seen[(int(p1), int(p2))] = (
            chargeProd.value_in_unit(unit.elementary_charge * unit.elementary_charge),
            sigma.value_in_unit(unit.nanometer),
            epsilon.value_in_unit(unit.kilojoule_per_mole),
        )

    assert seen[(0, 2)][0] == pytest.approx(0.0, abs=1e-12)
    assert seen[(1, 2)][0] == pytest.approx(0.0, abs=1e-12)
    assert seen[(0, 2)][1] == pytest.approx(expected_sigma_02, abs=1e-12)
    assert seen[(1, 2)][1] == pytest.approx(expected_sigma_12, abs=1e-12)
    assert seen[(0, 2)][2] == pytest.approx(expected_eps_02, abs=1e-12)
    assert seen[(1, 2)][2] == pytest.approx(expected_eps_12, abs=1e-12)


class _FakeState:
    def __init__(self, positions_angstrom):
        self._positions = np.asarray(positions_angstrom, dtype=np.float64) * unit.angstrom

    def getPositions(self, asNumpy=False):
        return self._positions


class _FakeModel:
    def __init__(self, dtype=torch.float32):
        self.r_max = torch.tensor(3.0, dtype=dtype)
        self.dtype = dtype

    def __call__(self, input_dict, compute_force=True):
        del compute_force
        n_ml = input_dict["positions"].shape[0]
        forces = torch.tensor(
            [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]][:n_ml],
            dtype=self.dtype,
            device=input_dict["positions"].device,
        )
        out = {
            "interaction_energy": torch.tensor([2.5], dtype=self.dtype, device=forces.device),
            "forces": forces,
        }
        if "mm_positions" in input_dict:
            out["mm_forces"] = torch.tensor(
                [[0.5, 0.25, -0.75]],
                dtype=self.dtype,
                device=forces.device,
            )
        return out


class PolarMACE:
    pass


class MACE:
    pass


def testComputeMACEScattersMMForces():
    ptr = torch.tensor([0, 2], dtype=torch.long)
    node_attrs = torch.ones((2, 1), dtype=torch.float32)
    batch = torch.zeros(2, dtype=torch.long)
    pbc = torch.tensor([False, False, False], dtype=torch.bool)
    mm_info = {"mm_atoms": np.array([2], dtype=np.int64), "mm_charges": np.array([0.25], dtype=np.float64)}

    energy, forces = _computeMACE(
        state=_FakeState([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        model=_FakeModel(dtype=torch.float32),
        ptr=ptr,
        node_attrs=node_attrs,
        batch=batch,
        pbc=pbc,
        returnEnergyType="interaction_energy",
        charge=torch.tensor([0.0], dtype=torch.float32),
        multiplicity=torch.tensor([1.0], dtype=torch.float32),
        indices=np.array([0, 1], dtype=np.int64),
        periodic=False,
        linkInfo=None,
        mmInfo=mm_info,
    )

    assert energy == pytest.approx(2.5 * 96.4853)
    expected = np.array(
        [
            [1.0, 2.0, 3.0],
            [-1.0, -2.0, -3.0],
            [0.5, 0.25, -0.75],
        ],
        dtype=np.float32,
    ) * (96.4853 * 10.0)
    np.testing.assert_allclose(forces, expected, rtol=1e-6, atol=1e-6)


def testShouldUseMMEmbedding():
    assert _should_use_mm_embedding(PolarMACE(), [0, 1], "electrostatic")
    assert not _should_use_mm_embedding(PolarMACE(), [0, 1], "mechanical")
    assert not _should_use_mm_embedding(PolarMACE(), None, "electrostatic")
    assert not _should_use_mm_embedding(MACE(), [0, 1], "electrostatic")
    with pytest.raises(ValueError, match="Unsupported embedding mode"):
        _should_use_mm_embedding(PolarMACE(), [0, 1], "bad-mode")

@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestMACE:
    @pytest.mark.parametrize("model", ['mace-off23-small', 'mace-off23-medium', 'mace-off23-large', 'mace-off24-medium',
                                       'mace-mpa-0-medium', 'mace-omat-0-small', 'mace-omat-0-medium', 'mace-omol-0-extra-large'])
    def testCreatePureMLSystem(self, platform_int, model):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential(model)
        system = potential.createSystem(pdb.topology, returnEnergyType='energy')
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.getPositions(asNumpy=True))
        energyML = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        # Reference energies are calculated with MACECalculator
        refEnergy = {'mace-off23-small': -713468.6327560507,
                     'mace-off23-medium': -713468.0563706581,
                     'mace-off23-large': -713467.7476380612,
                     'mace-off24-medium': -713467.9394350434,
                     'mace-mpa-0-medium': -8839.299589829867,
                     'mace-omat-0-small': -8726.63865431241,
                     'mace-omat-0-medium': -8679.026847088873,
                     'mace-omol-0-extra-large': -712903.4934289698}
        assert np.isclose(refEnergy[model], energyML, rtol=1e-6)

    def testPeriodicSystem(self, platform_int):
        pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        potential = MLPotential("mace-off23-small")
        system = potential.createSystem(pdb.topology, returnEnergyType='energy')
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        positionsOriginal = pdb.getPositions(asNumpy=True)
        energyRef = -151723354.26015 # Calculated with MACECalculator
        for i in range(3):
            positions = positionsOriginal + i * 0.9 * unit.nanometers # translate molecule to test PBC
            context.setPositions(positions)
            energyML = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            assert np.isclose(energyRef, energyML, rtol=1e-5)

    def testCreateMixedSystem(self, platform_int):
        prmtop = app.AmberPrmtopFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7"))
        inpcrd = app.AmberInpcrdFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7"))
        mlAtoms = list(range(15))
        mmSystem = prmtop.createSystem(nonbondedMethod=app.PME)
        potential = MLPotential("mace-off23-small")
        mixedSystem = potential.createMixedSystem(prmtop.topology, mmSystem, mlAtoms, interpolate=False)
        interpSystem = potential.createMixedSystem(prmtop.topology, mmSystem, mlAtoms, interpolate=True)
        platform = mm.Platform.getPlatform(platform_int)
        mmContext = mm.Context(mmSystem, mm.VerletIntegrator(0.001), platform)
        mixedContext = mm.Context(mixedSystem, mm.VerletIntegrator(0.001), platform)
        interpContext = mm.Context(interpSystem, mm.VerletIntegrator(0.001), platform)
        mmContext.setPositions(inpcrd.positions)
        mixedContext.setPositions(inpcrd.positions)
        interpContext.setPositions(inpcrd.positions)
        mmEnergy = mmContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        mixedEnergy = mixedContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        interpEnergy1 = interpContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        interpContext.setParameter('lambda_interpolate', 0)
        interpEnergy2 = interpContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        assert np.isclose(mixedEnergy, interpEnergy1, rtol=1e-5)
        assert np.isclose(mmEnergy, interpEnergy2, rtol=1e-5)


# ---------------------------------------------------------------------------
# Periodic link-atom (minimum-image cap placement) -- the openmm-ml fix that
# lets periodic/PME QM/MM builds with frontier-bond caps run.
# ---------------------------------------------------------------------------

def test_min_image_orthorhombic():
    cell = np.diag([10.0, 10.0, 10.0])
    v = np.array([[9.0, 0.0, 0.0],      # wraps to -1
                  [-0.5, 6.0, 0.0],     # y wraps to -4
                  [0.2, 0.3, -0.4]])    # already minimal
    out = _min_image(v, cell)
    np.testing.assert_allclose(
        out, [[-1.0, 0.0, 0.0], [-0.5, -4.0, 0.0], [0.2, 0.3, -0.4]], atol=1e-12
    )


def test_min_image_triclinic():
    cell = np.array([[10.0, 0.0, 0.0], [2.0, 9.0, 0.0], [1.0, 1.5, 8.0]])
    rem = np.array([[0.3, -0.2, 0.1]])
    v = rem + cell[0] + cell[2]         # one full lattice step (a + c) + remainder
    out = _min_image(v, cell)
    np.testing.assert_allclose(out, rem, atol=1e-10)


def _periodic_topology_system(n=3, box_nm=2.0):
    top = app.Topology()
    res = top.addResidue("X", top.addChain())
    for i in range(n):
        top.addAtom(f"A{i}", app.element.carbon, res)
    a = mm.Vec3(box_nm, 0, 0) * unit.nanometer
    b = mm.Vec3(0, box_nm, 0) * unit.nanometer
    c = mm.Vec3(0, 0, box_nm) * unit.nanometer
    top.setPeriodicBoxVectors((a, b, c))
    system = mm.System()
    for i in range(n):
        system.addParticle(12.0)
    system.setDefaultPeriodicBoxVectors(a, b, c)
    return top, system


def test_prepareLinkRecords_allows_periodic():
    # Regression for the lifted guard: a periodic system with link records used
    # to raise "linkRecords is only supported for non-periodic systems".
    top, system = _periodic_topology_system()
    info = _prepareLinkRecords([(0, 2, 1.09)], atoms=[0, 1], topology=top, system=system)
    assert info is not None and info["K"] == 1
    np.testing.assert_array_equal(info["q_global"], [0])
    np.testing.assert_array_equal(info["m_global"], [2])


class _FakePeriodicState(_FakeState):
    def __init__(self, positions_angstrom, box_angstrom):
        super().__init__(positions_angstrom)
        L = box_angstrom
        box = np.diag([L, L, L]) if np.isscalar(L) else np.asarray(L, dtype=np.float64)
        self._box = box * unit.angstrom

    def getPeriodicBoxVectors(self, asNumpy=False):
        return self._box


class _CapCapturingModel:
    """Records the positions handed to the model so a test can read the cap
    position (the last row, after the K appended caps)."""

    def __init__(self, dtype=torch.float64):
        self.r_max = torch.tensor(5.0, dtype=dtype)
        self.dtype = dtype
        self.seen_positions = None

    def __call__(self, input_dict, compute_force=True):
        del compute_force
        pos = input_dict["positions"]
        self.seen_positions = pos.detach().cpu().numpy().copy()
        n = pos.shape[0]
        return {
            "interaction_energy": torch.zeros(1, dtype=self.dtype, device=pos.device),
            "forces": torch.zeros((n, 3), dtype=self.dtype, device=pos.device),
        }


def _run_link(state, model, linkInfo, periodic):
    n_nodes = len(np.atleast_1d(linkInfo["q_global"])) + 2  # 2 ML atoms + K caps
    return _computeMACE(
        state=state, model=model,
        ptr=torch.tensor([0, n_nodes], dtype=torch.long),
        node_attrs=torch.ones((n_nodes, 1), dtype=torch.float64),
        batch=torch.zeros(n_nodes, dtype=torch.long),
        pbc=torch.tensor([periodic, periodic, periodic], dtype=torch.bool),
        returnEnergyType="interaction_energy",
        charge=torch.zeros(1, dtype=torch.float64),
        multiplicity=torch.ones(1, dtype=torch.float64),
        indices=np.array([0, 1], dtype=np.int64),
        periodic=periodic, linkInfo=linkInfo, mmInfo=None,
    )


def testComputeMACE_periodic_cap_uses_min_image():
    # Q (atom 0) at x=0.5, M (atom 2) at x=9.5 in a 10 A box: the true bond is the
    # 1 A image across the boundary (not the 9 A direct vector). With min-image,
    # C_L = 0.6/1, so the cap sits at Q + 0.6*(-1,0,0) = (-0.1, 5, 5).
    model = _CapCapturingModel()
    linkInfo = {"K": 1, "q_global": np.array([0]), "m_global": np.array([2]),
                "target_dist": np.array([0.6])}
    _run_link(_FakePeriodicState([[0.5, 5.0, 5.0], [2.0, 5.0, 5.0], [9.5, 5.0, 5.0]], 10.0),
              model, linkInfo, periodic=True)
    np.testing.assert_allclose(model.seen_positions[-1], [-0.1, 5.0, 5.0], atol=1e-9)


def testComputeMACE_nonperiodic_cap_unchanged():
    # Regression: non-periodic path is untouched -> direct interpolation.
    # M = atom 2 at (1,0,0), C_L = 0.6 -> cap at (0.6, 0, 0).
    model = _CapCapturingModel()
    linkInfo = {"K": 1, "q_global": np.array([0]), "m_global": np.array([2]),
                "target_dist": np.array([0.6])}
    _run_link(_FakeState([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
              model, linkInfo, periodic=False)
    np.testing.assert_allclose(model.seen_positions[-1], [0.6, 0.0, 0.0], atol=1e-9)


def testComputeMACE_periodic_bond_too_long_raises():
    # Min-image bond = (5,5,0), |.| = 7.07 A > inscribed-sphere radius 5 A of the
    # 10 A cube (a genuinely wrapped pair) -> unsupported, must raise.
    model = _CapCapturingModel()
    linkInfo = {"K": 1, "q_global": np.array([0]), "m_global": np.array([2]),
                "target_dist": np.array([0.6])}
    with pytest.raises(ValueError, match="inscribed-sphere radius"):
        _run_link(_FakePeriodicState([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 5.0, 0.0]], 10.0),
                  model, linkInfo, periodic=True)


def test_min_image_triclinic_skew_requires_lattice_shift():
    # codex counterexample: for a skewed cell the nearest image needs a lattice
    # shift even though every fractional component is < 0.5, so component rounding
    # alone is wrong. The 27-image refinement must recover v - a (the a-row).
    cell = np.array([[10.0, 0.0, 0.0], [4.9, 8.7, 0.0], [0.0, 0.0, 20.0]])
    v = np.array([[0.49, 0.46, 0.0]]) @ cell          # frac < 0.5 in every component
    out = _min_image(v, cell)
    np.testing.assert_allclose(out, v - cell[0], atol=1e-10)
    assert np.linalg.norm(out) < np.linalg.norm(v)    # genuinely shorter image


def testComputeMACE_periodic_force_redistribution_uses_min_image():
    # A nonzero force on the cap must be redistributed onto Q and M along the
    # minimum-image bond direction (not the raw 9 A vector).
    box, td = 10.0, 0.6
    f_cap = np.array([0.7, -0.3, 0.2])

    class _CapForceModel(_CapCapturingModel):
        def __call__(self, input_dict, compute_force=True):
            out = super().__call__(input_dict, compute_force)
            forces = out["forces"].clone()
            forces[-1] = torch.tensor(f_cap, dtype=self.dtype, device=forces.device)
            out["forces"] = forces
            return out

    model = _CapForceModel()
    linkInfo = {"K": 1, "q_global": np.array([0]), "m_global": np.array([2]),
                "target_dist": np.array([td])}
    _, forces = _run_link(
        _FakePeriodicState([[0.5, 5.0, 5.0], [2.0, 5.0, 5.0], [9.5, 5.0, 5.0]], box),
        model, linkInfo, periodic=True)
    # analytic redistribution with the min-image bond v = (-1,0,0), C_L = 0.6
    eS = 96.4853 * 10.0
    v = np.array([-1.0, 0.0, 0.0]); C_L = td / 1.0; e_b = v
    fc = f_cap * eS
    proj = fc @ e_b
    expected_Q = (1 - C_L) * fc + C_L * proj * e_b
    expected_M = C_L * fc - C_L * proj * e_b
    np.testing.assert_allclose(forces[0], expected_Q, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(forces[2], expected_M, rtol=1e-6, atol=1e-6)


def testComputeMACE_periodic_multiple_links_mixed_wrapped():
    # Two caps: one ordinary (atom 3 near atom 0) and one boundary-crossing
    # (atom 4 at x=9.6 vs Q atom 1 at x=0.4). Order and both placements preserved.
    box = 10.0
    pos = [[0.4, 5, 5], [0.4, 1, 5], [2.0, 5, 5],   # 0=Q_a, 1=Q_b(ml), 2 filler ml? no
           [1.4, 5, 5], [9.6, 1, 5]]                 # 3=M_a (direct), 4=M_b (wraps)
    # ML atoms = [0,1]; caps: (Q=0,M=3) ordinary, (Q=1,M=4) wrapped
    model = _CapCapturingModel()
    linkInfo = {"K": 2, "q_global": np.array([0, 1]), "m_global": np.array([3, 4]),
                "target_dist": np.array([0.6, 0.6])}
    n_nodes = 2 + 2
    _computeMACE(
        state=_FakePeriodicState(pos, box), model=model,
        ptr=torch.tensor([0, n_nodes], dtype=torch.long),
        node_attrs=torch.ones((n_nodes, 1), dtype=torch.float64),
        batch=torch.zeros(n_nodes, dtype=torch.long),
        pbc=torch.tensor([True, True, True]),
        returnEnergyType="interaction_energy",
        charge=torch.zeros(1, dtype=torch.float64),
        multiplicity=torch.ones(1, dtype=torch.float64),
        indices=np.array([0, 1], dtype=np.int64),
        periodic=True, linkInfo=linkInfo, mmInfo=None)
    caps = model.seen_positions[2:]                   # rows after the 2 ML atoms
    # cap a: Q0=(0.4,5,5), M3=(1.4,5,5) -> bond (1,0,0), C_L=0.6 -> (1.0,5,5)
    np.testing.assert_allclose(caps[0], [1.0, 5.0, 5.0], atol=1e-9)
    # cap b: Q1=(0.4,1,5), M4 image=(-0.4,1,5), bond (-0.8,0,0), C_L=0.6/0.8 -> (-0.2,1,5)
    np.testing.assert_allclose(caps[1], [-0.2, 1.0, 5.0], atol=1e-9)


def testComputeMACE_periodic_embedding_with_link_composes():
    # Electrostatic embedding (MM-charge forces) and a boundary-crossing link cap
    # in the same periodic call: the cap force must redistribute onto Q/M along
    # the minimum-image bond AND the MM force must scatter onto the MM atom (which
    # here is also the cap's M partner) -- the two contributions add on atom 2.
    box, td = 10.0, 0.6
    eS = 96.4853 * 10.0
    # positions: 0=Q(ml), 1=ml, 2=M(mm, wraps vs Q), 3=mm
    pos = [[0.5, 5, 5], [2.0, 5, 5], [9.5, 5, 5], [5.0, 5, 8]]

    class _EELinkModel:
        def __init__(self, dtype=torch.float64):
            self.r_max = torch.tensor(5.0, dtype=dtype)
            self.dtype = dtype

        def __call__(self, input_dict, compute_force=True):
            del compute_force
            dev = input_dict["positions"].device
            forces = torch.tensor([[1., 0., 0.], [0., 1., 0.], [3., 0., 1.]],
                                  dtype=self.dtype, device=dev)        # 2 ml + 1 cap
            out = {"interaction_energy": torch.zeros(1, dtype=self.dtype, device=dev),
                   "forces": forces}
            if "mm_positions" in input_dict:
                out["mm_forces"] = torch.tensor([[0.5, 0., 0.], [0., 0.5, 0.]],
                                                dtype=self.dtype, device=dev)
            return out

    linkInfo = {"K": 1, "q_global": np.array([0]), "m_global": np.array([2]),
                "target_dist": np.array([td])}
    mmInfo = {"mm_atoms": np.array([2, 3]), "mm_charges": np.array([-0.5, 0.3])}
    _, forces = _computeMACE(
        state=_FakePeriodicState(pos, box), model=_EELinkModel(),
        ptr=torch.tensor([0, 3], dtype=torch.long),
        node_attrs=torch.ones((3, 1), dtype=torch.float64),
        batch=torch.zeros(3, dtype=torch.long),
        pbc=torch.tensor([True, True, True]),
        returnEnergyType="interaction_energy",
        charge=torch.zeros(1, dtype=torch.float64),
        multiplicity=torch.ones(1, dtype=torch.float64),
        indices=np.array([0, 1], dtype=np.int64),
        periodic=True, linkInfo=linkInfo, mmInfo=mmInfo,
    )
    # analytic expectation (same min-image bond as cap placement)
    v = np.array([-1.0, 0.0, 0.0]); C_L = td / 1.0; e_b = v
    f_link = np.array([3.0, 0.0, 1.0]) * eS
    proj = f_link @ e_b
    F_Q = (1 - C_L) * f_link + C_L * proj * e_b
    F_M = C_L * f_link - C_L * proj * e_b
    exp = np.array([
        np.array([1.0, 0.0, 0.0]) * eS + F_Q,        # ml force + cap redistribution
        np.array([0.0, 1.0, 0.0]) * eS,              # ml force
        F_M + np.array([0.5, 0.0, 0.0]) * eS,        # cap redistribution + MM force
        np.array([0.0, 0.5, 0.0]) * eS,              # MM force
    ])
    np.testing.assert_allclose(forces, exp, rtol=1e-6, atol=1e-6)
