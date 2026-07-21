# Out-of-tree reproduction references

Checks that cannot run in the ordinary test suite because they need a model or
stack that is not an openmm-ml dependency.  Each one pairs a script that
produces a number with a committed text file holding the number it produced, so
a refactor can be checked against a real run without that stack being present.

## `polarmace_electrostatic_reference`

A single-point energy and force set for `embedding="electrostatic"`.  That
embedding needs a potential that accepts the charges and positions of the atoms
outside the ML subset; the only such model today is PolarMACE, which requires
mace-torch with the PolarMACE graft, `graph_longrange`, and a PolarMACE
checkpoint — none of them openmm-ml dependencies.

Regenerate, or check the current tree against the committed values:

```bash
python test/repro/polarmace_electrostatic_reference.py --model PolarMACE.model
python test/repro/polarmace_electrostatic_reference.py --model PolarMACE.model --check
```

The default force tolerance for `--check` is 1e-2 kJ/mol/nm, well above the
~1e-4 run-to-run scatter of the MACE forward on CPU and far below the
~2e3 kJ/mol/nm scale of the forces themselves.  The energy is reproducible to
the printed precision.
