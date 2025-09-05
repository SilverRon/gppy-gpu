# $\texttt{gppy-gpu}$: A GPU-Accelerated Imaging Pipeline for the 7-Dimensional Telescope and Sky Survey

> **Entrypoints**
> - **Watcher:** `run/routine/gpwatch_7DT_gain2750.py`  
> - **Main pipeline:** `run/routine/7DT_Routine_1x1_gain2750.py`

This repository provides an automated, production-grade imaging pipeline for The 7-Dimensional Telescope (7DT) and Sky Survey (7DS; [official website](http://7ds.snu.ac.kr/)). It covers **directory watching, calibration, astrometry (SCAMP), stacking (SWarp), PSF matching and image subtraction (HOTPANTS), transient detection (TBD), photometry, artifact masking, product curation, and Slack notifications**. When available, numerically intensive steps are **GPU-accelerated via CuPy**; CPU-only fallbacks are supported.

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

- **$\texttt{Source Extractor}$** (`sex` or `source-extractor`) [Bertin & Arnouts 1996](https://ui.adsabs.harvard.edu/abs/1996A%26AS..117..393B/abstract)  
- **$\texttt{SCAMP}$** [Bertin 2006](https://ui.adsabs.harvard.edu/abs/2006ASPC..351..112B/abstract)  
- **$\texttt{MissFITS}$** ([Terapix software page](https://www.astromatic.net/software/missfits/))  
- **$\texttt{Swarp}$** [Bertin et al. 2002](https://ui.adsabs.harvard.edu/abs/2002ASPC..281..228B/abstract)  
- **$\texttt{HOTPANTS}$** ([Becker 2015, GitHub](https://github.com/acbecker/hotpants))  
- **$\texttt{imhead}$** (e.g., from `wcstools`) for quick header snapshots  
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
python 7DT_Routine_1x1_gain2750.py 7DT07 /large_data/obsdata/7DT07/2024-04-23_gain2750
```

- `arg1`: observatory code (e.g., `7DT07`)  
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

## Data Management & Folder Structure

The 7DT/7DS pipeline is designed to interface with a standardized data hierarchy on the `proton` server and associated factory directories. This ensures reproducibility and traceability from raw acquisition to calibrated products.

### Raw Data
Located under `/large_data/obsdata/`, with observatory codes and nightly subfolders:
```
/large_data/obsdata/
├── 7DT01
│   ├── 2023-10-09
│   ├── 2023-10-10
│   ├── 2023-10-11
├── 7DT02
├── 7DT03
...
```

### Master Frames
Bias, dark, and flat masters are stored separately by unit:
```
/large_data/factory/master_frame/
├── 7DT01
│   ├── dark
│   ├── flat
│   └── zero
├── 7DT02
│   ├── dark
│   ├── flat
│   └── zero
...
```

### Reference Catalogs
```
/large_data/factory/ref_cat/
```

- Gaia XP continuous spectra (e.g., `XP_CONTINUOUS_RAW_<OBJECT>`.csv)
- Synthetic photometry (e.g., `gaiaxp_dr3_synphot_<OBJECT>.csv`)

### Calibrated Products
Processed data are stored under:
```
/large_data/processed_1x1_gain2750/
├── UDS
│   ├── 7DT01
│   │   ├── m400/ ── calib*.fits, calib*com.fits
│   │   ├── m425/phot/ ── calib*.phot.cat
│   │   └── m675/
│   ├── 7DT02
│   │   ├── m450/, m475/, m700/
│   ├── 7DT03
│   │   ├── m500/, m525/, m725/
...
```

- Single and stacked frames (`calib*.fits`, `calib*com.fits`)
- Photometry catalogs (`*.phot.cat`) per filter

### Outputs

The pipeline produces standardized image and catalog products:
- Calibrated images

  ```
  calib_<unit>_<field>_<date>_<time>_<filter>_<exptime>.fits
  ```

  Example: `calib_7DT01_UDS_20231105_003915_m400_60.fits`

- Photometry catalogs (`*.phot.cat`)
  Readable via `astropy.table`:

  ```
  from astropy.table import Table
  tbl = Table.read("calib_...phot.cat", format="ascii")
  ```

  Key columns include:
  - `MAG_APER`, `MAGERR_APER`: instrumental magnitudes and errors
  - `MAG_APER_m650`: ZP-corrected magnitude
  - `FLUX_APER[_m650]`, `SNR_APER_m650`: fluxes and SNR

- Apertures (suffix convention):
  - `AUTO`: SExtractor auto magnitude
  - `APER`: seeing × 2 × 0.673
  - `APER_1`: seeing × 2
  - `APER_2`: seeing × 3
  - `APER_3`: fixed 3″
  - `APER_4`: fixed 5″
  - `APER_5`: fixed 10″

Example usage:
  ```
  # From Header
  gethead calib_*.fits ZP_1
  ```

  ```
  # From Table
  print(tbl['MAG_APER_1_m650'])
  ```



## Authors & Maintainers

If you are interested in `7DT`/`7DS`, or need support, please reach out.

| Name                    | GitHub            | Role                                                                                           | Affiliation                                              | ORCID                                | Contact                                   |
|-------------------------|-------------------|--------------------------------------------------------------------------------------------------|----------------------------------------------------------|---------------------------------------|-------------------------------------------|
| **Gregory S. H. Paek**  | @SilverRon        | **Conceptualization; Software; Architecture; Methodology; Data curation; Documentation; Founding developer; Maintainer (v1.x)** | University of Hawaiʻi in Manoa, Institute for Astronomy (IfA); formerly Seoul National University (SNU) | `https://orcid.org/0000-0002-6639-6533` | `gregorypaek94 [at] gmail [dot] com`      |
| **Donghwan Hyun**       | @renormalization2 | Software (stacking-module refactoring); Code maintenance; Packaging; **Co-maintainer (v2.0)**                 | Seoul National University (SNU)                           | `https://orcid.org/0009-0009-4501-5285` | hdhd333 [at] gmail [dot] com                                         |
| **Won-Hyeong Lee**      | @Yicircle         | Software; Integration; Testing; **Co-maintainer (v2.0)**                                                      | Seoul National University (SNU)                           | `https://orcid.org/0009-0005-6140-8303` | wohy1220 [at] gmail [dot] com                                         |
| **Myungshin Im**        | —                 | Supervision; Funding acquisition; Investigation; Resources                                                    | Seoul National University (SNU)                           | `https://orcid.org/0000-0002-8537-6714` | myungshin.im [at] gmail [dot] com                                         |
<!-- | **Ji Hoon Kim**         | —                 | Project administration; Resources; Investigation                                                              | Seoul National University (SNU)                           | `https://orcid.org/XXXX-XXXX-XXXX-XXXX` | —                                         | -->

**Corresponding / Maintainer:** Gregory S. H. Paek (`gregorypaek94 [at] gmail [dot] com`)

### Project History & Handover

- **v1.x (foundational)** — Designed and implemented by **Gregory S. H. Paek** (sole author of the initial framework and architecture).  
- **Refactoring & transition** — **Donghwan Hyun** contributed targeted **refactoring of the stacking modules** and participated in the handover.  
- **v2.0 (in progress)** — Developed **independently (separate repository)** by **Donghwan Hyun** following the handover. While inheriting most of the functionality and philosophy of **v1.x**, this version introduces **structural changes for parallel execution**, integrates support for **various weights and error maps**, and updates for **seamless database connectivity**. This repository continues to track the stable **v1.x** line.

### Contributors

- @SilverRon — **Gregory S. H. Paek**  
- @renormalization2 — **Donghwan Hyun**  
- @Yicircle — **Won-Hyeong Lee**

## Suggested Reading

- **7-Dimensional Sky Survey (7DS)**  
  *Im, Myungshin* — *43rd COSPAR Scientific Assembly*, Jan 2021.  
  **Citation:** Im, M. (2021), *43rd COSPAR Scientific Assembly*, E1537.  
  **ADS:** https://ui.adsabs.harvard.edu/abs/2021cosp...43E1537I/abstract
  
- **Introduction to the 7-Dimensional Telescope: commissioning procedures and data characteristics**  
  *Kim, Ji Hoon; Im, Myungshin; Lee, Hyungmok; Chang, Seo-Won; Choi, Hyeonho; Paek, Gregory S. H.* — *Proceedings of SPIE, Volume 13094, id. 130940X* (Aug 2024).  
  **Citation:** Kim, J. H., Im, M., Lee, H., Chang, S.-W., Choi, H., & Paek, G. S. H. (2024), *Proc. SPIE 13094, id. 130940X, 11 pp.*  
  **DOI:** [10.1117/12.3019546](https://doi.org/10.1117/12.3019546)  
  **ADS:** [2024SPIE13094E..0XK](https://ui.adsabs.harvard.edu/abs/2024SPIE13094E..0XK/abstract)


- **TCSpy: Multiple Telescope Control System for 7 Dimensional Telescope (7DT)**  
  *Choi, Hyeonho; Im, Myungshin; Kim, Ji Hoon* — *IAU General Assembly*, Aug 2024.  
  **Citation:** Choi, H., Im, M., & Kim, J. H. (2024), *IAU General Assembly*, 32, P1281.  
  **ADS:** https://ui.adsabs.harvard.edu/abs/2024IAUGA..32P1281C/abstract



### Acknowledgements

...
<!-- This repository extends and refines code originally developed by **Dr. Yujin Yang (KASI, 2022)**. We gratefully acknowledge her contributions and the broader 7DT/7DS collaboration. -->

<!-- > **Notes**
> - Roles follow the **CRediT** taxonomy for academic clarity.  
> - Consider adding a `CITATION.cff` file and updating ORCID links when available.  
> - To prevent leaking secrets, ensure `/config/keys.dat` is listed in `.gitignore`. -->
