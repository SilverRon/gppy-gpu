# gppy-gpu — A GPU-Accelerated Imaging Pipeline for the 7-Dimensional Telescope (7DT) and Sky Survey (7DS)

> **Entrypoints**
> - **Watcher:** `run/routine/gpwatch_7DT_gain2750.py`  
> - **Main pipeline:** `run/routine/7DT_Routine_1x1_gain2750.py`

This repository provides an automated, production-grade imaging pipeline for the 7-Day Transient (7DT) program. It covers **directory watching, calibration, astrometry (SCAMP), stacking, PSF matching and image subtraction (HOTPANTS), transient detection & photometry, artifact masking, product curation, and Slack notifications**. When available, numerically intensive steps are **GPU-accelerated via CuPy**; CPU-only fallbacks are supported.

The codebase is intended for unattended operation in observatory environments where new data arrive in episodic batches, and can also be run **ad hoc** on specific nights or targets.

---

## Key Features

- **Robust directory watcher:** monitors incoming observation folders and triggers processing once size growth stabilizes.  
- **Astrometry:** Source Extractor (pre-catalog) → SCAMP (WCS) → MissFITS (header application).  
- **Stacking:** per target/filter/time block with effective EGAIN and provenance recorded in headers.  
- **PSF matching & subtraction:** HOTPANTS with tiled kernels, configurable thresholds, and mask propagation.  
- **Transient detection & photometry:** catalogs (`*.cat`), regions (`*.reg`), and diagnostic PNGs for single and stacked frames.  
- **Product curation:** reproducible directory structure under a configurable processed root; header snapshots archived.  
- **Notifications:** start/finish (and failure) messages to Slack.  
- **Parallelism & GPU offload:** multi-core CPU parallelization with optional CUDA acceleration via CuPy.  

---

## Repository Layout (abridged)

```text
gppy-gpu/
├─ run/
│  └─ routine/
│     ├─ gpwatch_7DT_gain2750.py          # watcher (entry)
│     ├─ 7DT_Routine_1x1_gain2750.py      # main pipeline (entry)
│     ├─ path.json                        # required path config
│     ├─ gppy_tmux.sh                     # optional tmux launcher for multi-watch
│     └─ (stack helpers, if any)
├─ config/
│  ├─ *.sex / *.param / *.conv / *.nnw    # Source Extractor configs
│  ├─ 7dt.scamp                           # SCAMP config
│  ├─ obs.dat, ccd.dat, fringe.dat        # site/CCD/fringe metadata
│  ├─ alltarget.dat, changehdr.dat        # target list, header rewrite rules
│  └─ keys.dat                            # tokens/credentials (DO NOT COMMIT)
├─ src/
│  ├─ phot/                               # photometry & detection utilities
│  │  ├─ gregoryphot_7DT_NxN.py
│  │  └─ gregorydet_7DT_NxN.py
│  ├─ util/
│  │  ├─ gregorysubt_7DT.py               # HOTPANTS wrapper & masks
│  │  ├─ tool.py, path_manager.py         # logging, Slack, path helpers
│  └─ preprocess/calib.py                  # master frames, file normalization
└─ LICENSE, README.md, requirements.txt, ...
```

> If you restructure `src/`, adjust imports in the two entry scripts accordingly.

---

## System Requirements

### External Binaries

- **Source Extractor** (`sex` or `source-extractor`)
- **SCAMP** and **MissFITS**
- **HOTPANTS**
- **imhead** (e.g., from `wcstools`) for quick header snapshots
- (Optional) **SAOImage DS9** for visual inspection of `*.reg` overlays

Ensure these tools are discoverable on your `PATH`.

### Python Environment

- Python ≥ 3.10 (3.11 recommended)  
- See `requirements.txt` for libraries. Minimal core includes:
  - `numpy`, `astropy`, `scipy`, `matplotlib`
  - `cupy` *(CUDA-enabled; version must match system CUDA)* — optional
  - `tqdm`, `psutil`, `PyYAML`, `requests` (for Slack)
  - any site-specific wrapper utilities

**Create environment (example):**
```bash
mamba create -n gppy-gpu python=3.11 -y
mamba activate gppy-gpu
pip install -r requirements.txt
# If using GPU:
# pip install cupy-cuda12x  # choose the build matching your CUDA toolkit
```

---

## Configuration

### `run/routine/path.json` (required)

```json
{
  "path_base": "/large_data/factory",
  "path_obsdata": "",
  "path_processed": "",
  "path_refcat": "",
  "path_ref_scamp": "",
  "path_log": ""
}
```

- Defaults if left empty:
  - `path_obsdata`: `<path_base>/../obsdata`  
  - `path_processed`: `<path_base>/../processed_1x1_gain2750`  
  - `path_refcat`: `<path_base>/ref_cat`  
  - `path_ref_scamp`: `<path_base>/ref_scamp`  
  - `path_log`: `<path_base>/log/<obs>.log`  
- All paths must be readable/writable by the pipeline user.

### Credentials

`config/keys.dat` stores tokens (e.g., Slack). **Do not commit secrets.** Example:

```text
name   key                                      pw
slack  xoxb-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX    ""
TNS    <your-tns-token>                         ""
```

> To prevent accidental commits, add `/config/keys.dat` to `.gitignore`.

### Reference Frames

Place references under:

```text
<path_base>/ref_frame/<filter>/
  ref_PS1_<OBJ>_00000000_000000_<filter>_0.fits       # Pan-STARRS reference
  ref_7DT_<OBJ>_<...>_<filter>_<...>.fits             # optional in-house reference
```

The main pipeline prefers 7DT references if present, otherwise falls back to PS1.

---

## Input Data & Naming Conventions

The watcher scans `"<path_obsdata>/<OBS>/"` for new night folders. By default it recognizes:

- Regular monitoring: `YYYY-MM-DD_gain2750`  
- ToO/test: additional patterns configured in the watcher script (e.g., `pattern_too`, `pattern_test`)

A folder is considered “ready” when its size remains unchanged for a grace period; the watcher then triggers the pipeline.

---

## Usage

### A. One-off processing (main pipeline)

```bash
cd run/routine
python 7DT_Routine_1x1_gain2750.py 7DT03 /large_data/obsdata/7DT03/2024-04-23_gain2750
```

- `arg1`: observatory code (e.g., `7DT03`)  
- `arg2`: target night directory. If omitted, the script can search for the most recent stabilized folder.

### B. Continuous operations (watcher)

```bash
cd run/routine
python gpwatch_7DT_gain2750.py 7DT07
```

Run multiple watchers (one per site) under tmux:

```bash
bash run/routine/gppy_tmux.sh  # edit the observations array inside if needed
```

---

## Processing Stages (high-level)

1. **Init & logging**  
   Load `path.json`, site/CCD/target metadata (`config/*.dat`), and Slack credentials; emit a “start” message.

2. **Master frames (GPU-accelerated)**  
   Build bias/dark/flat masters (CuPy when available). Select the best-matching dark exposure time and apply scaling.

3. **Pre-detection & astrometry**  
   Run Source Extractor with `config/*.sex/param/conv/nnw` to generate pre-cata*

## Authors & Maintainers

If you are interested in `7DT`/`7DS`, or need support, please reach out.

| Name                    | GitHub            | Role (CRediT-style)                                                                                           | Affiliation                                              | ORCID                                | Contact                                   |
|-------------------------|-------------------|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|---------------------------------------|-------------------------------------------|
| **Gregory S. H. Paek**  | @SilverRon        | **Conceptualization; Software; Architecture; Methodology; Data curation; Documentation; Founding developer; Maintainer (v1.x)** | University of Hawaiʻi, Institute for Astronomy (IfA); formerly SNU | `https://orcid.org/XXXX-XXXX-XXXX-XXXX` | `gregorypaek94 [at] gmail [dot] com`      |
| **Donghwan Hyun**       | @renormalization2 | Software (stacking-module refactoring); Code maintenance; Packaging; **Co-maintainer (v2.0)**                 | Seoul National University (SNU)                           | `https://orcid.org/XXXX-XXXX-XXXX-XXXX` | —                                         |
| **Won-Hyeong Lee**      | @Yicircle         | Software; Integration; Testing; **Co-maintainer (v2.0)**                                                      | Seoul National University (SNU)                           | `https://orcid.org/XXXX-XXXX-XXXX-XXXX` | —                                         |
| **Myungshin Im**        | —                 | Supervision; Funding acquisition; Investigation; Resources                                                    | Seoul National University (SNU)                           | `https://orcid.org/XXXX-XXXX-XXXX-XXXX` | —                                         |
| **Ji Hoon Kim**         | —                 | Project administration; Resources; Investigation                                                              | Seoul National University (SNU)                           | `https://orcid.org/XXXX-XXXX-XXXX-XXXX` | —                                         |

**Corresponding / Maintainer:** Gregory S. H. Paek (`gregorypaek94 [at] gmail [dot] com`)

### Project History & Handover

- **v1.x (foundational)** — Designed and implemented by **Gregory S. H. Paek** (sole author of the initial framework and architecture).  
- **Refactoring & transition** — **Donghwan Hyun** contributed targeted **refactoring of the stacking modules** and participated in the handover.  
- **v2.0 (in progress)** — Currently being developed **independently (separate repository)** by **Donghwan Hyun** and **Won-Hyeong Lee** following the handover, while this repository tracks the stable **v1.x** line.

### Contributors

- @SilverRon — **Gregory S. H. Paek**  
- @renormalization2 — **Donghwan Hyun**  
- @Yicircle — **Won-Hyeong Lee**

### Acknowledgements

This repository extends and refines code originally developed by **Dr. Yujin Yang (KASI, 2022)**. We gratefully acknowledge her contributions and the broader 7DT/7DS collaboration.

<!-- > **Notes**
> - Roles follow the **CRediT** taxonomy for academic clarity.  
> - Consider adding a `CITATION.cff` file and updating ORCID links when available.  
> - To prevent leaking secrets, ensure `/config/keys.dat` is listed in `.gitignore`. -->
