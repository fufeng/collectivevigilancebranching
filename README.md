# collectivevigilancebranching

# Figure 1 code

Four self-contained Python scripts reproduce the individual panels of the finalized collective-vigilance figure.

| Script | Panel |
| --- | --- |
| `fig1A.py` | Analytical stability and Nash-equilibrium map |
| `fig1B.py` | Basins of attraction for behavioral adjustment |
| `fig1C.py` | Individual vigilance trajectories |
| `fig1D.py` | Evolution of vigilance under a Wright–Fisher model |

## Run

Use Python 3.10 or later. Install the dependencies (versions used in the archived release):

```sh
python -m pip install numpy==2.2.6 matplotlib==3.10.7 Pillow==12.0.0
python fig1A.py
python fig1B.py
python fig1C.py
python fig1D.py
```

Outputs are written to `fig1_output/` beside the scripts: PDF, SVG, 600-dpi PNG, preview PNG, and numerical/export checks for each panel. Add TIFF or change the destination with, for example:

```sh
python fig1A.py --formats pdf svg png tiff --output-dir figures
```

Panel A is analytical. Panels B–D load the bundled `fig1_data/panel_B.npz`, `panel_C.npz`, and `panel_D.npz`. Keep that folder beside the scripts when sharing them. Each script contains its own model and plotting code and does not import the other scripts or the original project.

## Optional: regenerate simulation data

```sh
python fig1B.py --simulate
python fig1C.py --simulate
python fig1D.py --simulate
```

These commands overwrite the corresponding bundled dataset before plotting; use `--data-dir regenerated_data` to save elsewhere. Full simulations can take several minutes. Model parameters are defined in `P` near the top of each script. If you change them, regenerate the data. Panel D uses seed 0; bundled data preserve the archived realization. Some validation and annotations assume the published parameters.

The scripts preserve the model, data, dimensions, and drawing settings from `release/collective-vigilance-figure/`. Arial is used when installed, with DejaVu Sans as fallback.
