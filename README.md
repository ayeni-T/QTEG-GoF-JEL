# QTEG GoF: Characterisation-Based JEL Goodness-of-Fit Test

Replication code for the paper:

> **"A Characterisation-Based Jackknife Empirical Likelihood Goodness-of-Fit Test for the Quadratic-Transformed Exponential-Gamma Distribution"**  
> Taiwo Michael Ayeni and Yichuan Zhao  
> Department of Mathematics and Statistics, Georgia State University, 2026

---

## Overview

This repository contains all simulation and plotting code used to produce the results in the paper. The proposed test is a characterisation-based goodness-of-fit procedure for the QTEG distribution, derived from the Lukacs independence property of the Gamma family. The JEL statistic follows a χ²₁ limiting distribution under the null, requiring no parametric bootstrap calibration. A Bartlett-corrected variant (BC-JEL) is also implemented. Comparison tests, Kolmogorov–Smirnov (KS), Anderson–Darling (AD), and Cramér–von Mises (CvM), are calibrated via parametric bootstrap on identical simulated samples.

---

## Requirements

```
Python >= 3.9
numpy
scipy
matplotlib
pandas
```

Install dependencies:

```bash
pip install numpy scipy matplotlib pandas
```

---

## Simulation Design

| Component | Detail |
|---|---|
| Replications | N = 5,000 per block |
| Sample sizes | n ∈ {30, 50, 100, 200} |
| Null scenarios | Sc.1: QTEG(1.5, 0.5), Sc.2: QTEG(2.0, 1.0), Sc.3: QTEG(3.0, 2.0) |
| Alternatives | Weibull(0.8), Weibull(1.5), LogNormal(0,1), Exponential(1), Gamma(2,1) |
| Bootstrap samples | B = 300 per replication (EDF calibration) |
| Total SLURM blocks | 72 (blocks 0–11: size; blocks 12–71: power) |
| Nominal level | δ = 5% |

All tests are evaluated on **identical simulated samples** per replication to ensure a fair comparison.

---

## Usage

### 1. Run a quick test (3 replications, B = 10)

```bash
python QTEG_GoF_Arctic.py --test
```

### 2. Run a single block locally

```bash
python QTEG_GoF_Arctic.py --block 0 --n_sim 5000 --B 300
```

Block numbering:
- Blocks 0–11: size study (3 null scenarios × 4 sample sizes)
- Blocks 12–71: power study (3 null scenarios × 5 alternatives × 4 sample sizes)

### 3. Submit all 72 blocks on HPC (SLURM)

```bash
sbatch qteg_gof_array.sh
```

Submit only new power blocks (if size blocks already complete):

```bash
sbatch --array=12-71 qteg_gof_array.sh
```

### 4. Merge results after all blocks complete

```bash
python QTEG_GoF_Arctic.py --merge
```

This produces `results/QTEG_GoF_full_results.csv` and `results/QTEG_GoF_full_results.json`.

### 5. Run real data analysis

```bash
python QTEG_GoF_Arctic.py --realdata
```

Four datasets are included internally:
- DS1: Bladder cancer remission times (n = 128)
- DS2: Boeing 720 aircraft failure times (n = 213)
- DS3: Malignant melanoma survival times (n = 205)
- DS4: Guinea pig survival under tubercle bacilli (n = 72)

### 6. Generate all paper figures

```bash
python QTEG_GoF_plot_results.py
```

Figures saved to `results/` as both `.png` and `.pdf`:

| File | Description |
|---|---|
| `fig1_size_all` | Empirical size, all 5 tests, 3 null scenarios |
| `fig2_power_main` | Power curves, Weibull(1.5), LogNormal(0,1), Exponential(1) |
| `fig3_power_hard` | Power curves, Gamma(2,1), Weibull(0.8) (near-null) |
| `fig4_power_summary_n200` | Bar chart, power at n = 200, all alternatives |
| `fig5_sqrtY_overlay` | KDE vs fitted Gamma density (real data) |
| `fig6_pp_plots` | PP plots with 95% KS band (real data) |
| `fig_supp_null_calibration` | Mean JEL statistic converging to χ²₁ mean |

> **Note for HPC users:** Update `--account`, `--mail-user`, and all path
> placeholders in `qteg_gof_array.sh` before submitting.

---

## Reproducibility

- Base random seed: `20260501`
- Each block uses seed `BASE_SEED + block_id × 10000 + rep`
- Bootstrap seeds are offset by `+1,000,000` from the replication seed
- All results are checkpointed every 50 replications to `results/gof_block_XX_partial.json`

---

## Citation

If you use this code, please cite:

```
Ayeni, T. M. and Zhao, Y. (2026). A Characterisation-Based Jackknife Empirical
Likelihood Goodness-of-Fit Test for the Quadratic-Transformed Exponential-Gamma
Distribution. Manuscript in preparation.
```

---

## Authors

**Taiwo Michael Ayeni**, Department of Mathematics and Statistics, Georgia State University  
**Yichuan Zhao**, Department of Mathematics and Statistics, Georgia State University
