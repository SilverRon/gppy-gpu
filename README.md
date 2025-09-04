# gppy-gpu — A GPU-Accelerated Imaging Pipeline for the 7-Day Transient (7DT) Survey

> **Entrypoints**
> - **Watcher:** `run/routine/gpwatch_7DT_gain2750.py`  
> - **Main pipeline:** `run/routine/7DT_Routine_1x1_gain2750.py`

This repository implements an automated, production-oriented pipeline for the 7-Day Transient (7DT) program, covering **directory watching, calibration, astrometric solution (SCAMP), stacking, PSF matching and image subtraction (HOTPANTS), source detection & photometry, artifact masking, product curation, and Slack notifications**. Where available, numerically intensive steps are **GPU-accelerated via CuPy**; CPU-only fallbacks are possible.

The codebase is designed for unattended operations in observatory environments where new data arrive in episodic batches, but it can also be run **ad hoc** on specific nights or targets.

---

## Key Features

- **Robust directory watcher**: monitors incoming observation folders; triggers the pipeline once size growth stabilizes.
- **Astrometry**: Source Extractor (pre-catalog) → SCAMP (WCS) → MissFITS (header application).
- **Stacking**: per target/filter/time block; effective EGAIN and provenance recorded in headers.
- **PSF matching & subtraction**: HOTPANTS with tiled kernels; configurable thresholds; mask propagation.
- **Transient detection & photometry**: catalogs (`*.cat`), regions (`*.reg`), and diagnostic PNGs for single and stacked frames.
- **Product curation**: reproducible directory structure under a configurable processed root; header snapshots archived.
- **Notifications**: start/finish (and failure) messages to Slack.
- **Parallelism & GPU offload**: multi-core CPU parallelization with optional CUDA acceleration via CuPy.

---

## Repository Layout (abridged)
```gppy-gpu/
├─ run/
│ └─ routine/
│ ├─ gpwatch_7DT_gain2750.py # watcher (entry)
│ ├─ 7DT_Routine_1x1_gain2750.py # main pipeline (entry)
│ ├─ path.json # required path config
│ ├─ gppy_tmux.sh # optional tmux launcher for multi-watch
│ └─ (stack helpers, if any)
├─ config/
│ ├─ *.sex / *.param / *.conv / *.nnw # Source Extractor configs
│ ├─ 7dt.scamp # SCAMP config
│ ├─ obs.dat, ccd.dat, fringe.dat # site/CCD/fringe metadata
│ ├─ alltarget.dat, changehdr.dat # target list, header rewrite rules
│ └─ keys.dat # tokens/credentials (DO NOT COMMIT)
├─ src/
│ ├─ phot/ # photometry & detection utilities
│ │ ├─ gregoryphot_7DT_NxN.py
│ │ └─ gregorydet_7DT_NxN.py
│ ├─ util/
│ │ ├─ gregorysubt_7DT.py # HOTPANTS wrapper & masks
│ │ ├─ tool.py, path_manager.py # logging, Slack, path helpers
│ └─ preprocess/calib.py # master frames, file normalization
└─ LICENSE, README.md, requirements.txt, ...
```


> The exact module names above reflect the current organization; if you restructure `src/`, adjust imports in the two entry scripts accordingly.

---

## System Requirements

### External Binaries
- **Source Extractor** (`sex` or `source-extractor`)
- **SCAMP** and **MissFITS**
- **HOTPANTS**
- **imhead** (e.g., from `wcstools`) for quick header snapshots
- (Optional) **SAOImage DS9** for visual inspection of `*.reg` overlays

Ensure the above are discoverable on `PATH`.

### Python Environment

- Python ≥ 3.10 (3.11 recommended)
- See `requirements.txt` for libraries. Minimal core:
  - `numpy`, `astropy`, `scipy`, `matplotlib`
  - `cupy` *(CUDA-enabled; version must match system CUDA)* — optional
  - `tqdm`, `psutil`, `PyYAML`, `requests` (for Slack)
  - any wrapper utilities used by your local deployment

**Create environment (example):**
```bash
mamba create -n gppy-gpu python=3.11 -y
mamba activate gppy-gpu
pip install -r requirements.txt
# If using GPU:
# pip install cupy-cuda12x  # match x to your CUDA minor
```

### Configuration
`run/routine/path.json` (required)
{
  "path_base": "/large_data/factory",
  "path_obsdata": "",           // default: <path_base>/../obsdata
  "path_processed": "",         // default: <path_base>/../processed_1x1_gain2750
  "path_refcat": "",            // default: <path_base>/ref_cat
  "path_ref_scamp": "",         // default: <path_base>/ref_scamp
  "path_log": ""                // default: <path_base>/log/<obs>.log
}

Leave optional entries empty to use the indicated defaults. All paths must be readable/writable by the pipeline user.
config/keys.dat (sensitive; do not commit)
Tab- or space-delimited keyring for services such as Slack:

SLACK_TOKEN	<your_slack_token>
SLACK_CHANNEL	<your_slack_channel>