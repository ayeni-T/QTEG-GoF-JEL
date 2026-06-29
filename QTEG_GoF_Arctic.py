"""
================================================================================
QTEG_GoF_Arctic_v4.py  --  JEL + Bootstrap-Calibrated EDF (unified)
================================================================================
Runs JEL, KS, AD, and CvM tests on EVERY replication using the SAME sample.
Bootstrap B=300 parametric samples used to calibrate KS/AD/CvM under the
composite QTEG null.

Changes from v3:
  - Added edf_statistics() and bootstrap_edf_pvalues()
  - run_gof_test() now returns JEL + KS + AD + CvM results
  - Simulation counters extended: rej_ks, rej_ad, rej_cvm
  - _make_summary() extended with rej_ks, se_ks, rej_ad, se_ad,
    rej_cvm, se_cvm
  - CSV/JSON output includes all four tests
  - Everything else unchanged (same kernel, same JEL, same blocks)

All 72 blocks run in parallel via SLURM array.
Blocks 0-11:  size study (3 null scenarios x 4 sample sizes)
Blocks 12-71: power study (3 null scenarios x 5 alternatives x 4 sample sizes)
Heaviest block: n=200, ~17.5h wall time (within 48h qCPU120 limit).

Authors: Taiwo Michael Ayeni and Yichuan Zhao
         Georgia State University, 2026
================================================================================
"""

import numpy as np
from scipy.special import gammaln, gammainc
from scipy.optimize import minimize
from scipy.stats import chi2
import json, os, argparse, time, csv, re, math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)

# ── Simulation parameters ──────────────────────────────────────────────────────
N_SIM_DEFAULT    = 5000
B_BOOTSTRAP      = 300       # bootstrap samples for EDF calibration
BASE_SEED        = 20260501
CHECKPOINT_EVERY = 50
NOMINAL          = 0.05

NULL_SCENARIOS = [(1.5, 0.5), (2.0, 1.0), (3.0, 2.0)]
SAMPLE_SIZES   = [30, 50, 100, 200]

def _weibull(k):    return lambda n, rng: rng.weibull(k, n)
def _lognormal():   return lambda n, rng: rng.lognormal(0.0, 1.0, n)
def _exponential(): return lambda n, rng: rng.exponential(1.0, n)
def _gamma():       return lambda n, rng: rng.gamma(2.0, 1.0, n)

ALTERNATIVES = [
    ("Weibull(0.8)",   _weibull(0.8)),
    ("Weibull(1.5)",   _weibull(1.5)),
    ("LogNormal(0,1)", _lognormal()),
    ("Exponential(1)", _exponential()),
    ("Gamma(2,1)",     _gamma()),
]

N_SIZE_BLOCKS  = len(NULL_SCENARIOS) * len(SAMPLE_SIZES)                    # 12
N_POWER_BLOCKS = len(NULL_SCENARIOS) * len(ALTERNATIVES) * len(SAMPLE_SIZES)  # 60
N_TOTAL_BLOCKS = N_SIZE_BLOCKS + N_POWER_BLOCKS                               # 72

# ── QTEG functions ─────────────────────────────────────────────────────────────
def qteg_logpdf(y, alpha, beta):
    return (alpha*np.log(beta) - np.log(2.0) - gammaln(alpha)
            + ((alpha-2.0)/2.0)*np.log(y) - beta*np.sqrt(y))

def qteg_cdf(y, alpha, beta):
    return gammainc(alpha, beta*np.sqrt(np.maximum(y, 0.0)))

def qteg_sample(alpha, beta, n, rng):
    return rng.gamma(alpha, 1.0/beta, n)**2

# ── MLE ────────────────────────────────────────────────────────────────────────
def qteg_mle(y):
    y  = np.asarray(y, float)
    sy = np.mean(np.sqrt(y))
    def neg_ll(params):
        a, b = params
        if a <= 0 or b <= 0: return np.inf
        return -np.sum(qteg_logpdf(y, a, b))
    best_res, best_val = None, np.inf
    for a0 in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]:
        b0 = a0 / max(sy, 1e-8)
        try:
            res = minimize(neg_ll, [a0, b0], method='L-BFGS-B',
                           bounds=[(1e-6, None), (1e-6, None)],
                           options={'ftol': 1e-13, 'gtol': 1e-9, 'maxiter': 5000})
            if res.fun < best_val:
                best_val, best_res = res.fun, res
        except Exception:
            pass
    if best_res is None: return None
    return dict(alpha=float(best_res.x[0]), beta=float(best_res.x[1]),
                logL=-best_val, n=len(y))

# ── Corrected kernel (v2, unchanged) ──────────────────────────────────────────
def c_alpha_val(alpha):
    return 1.0 / (4.0 * (2.0*alpha + 1.0))

def _kernel_matrix(y, Vbar, c_alpha):
    w  = np.sqrt(y)
    wi = w[:, None]; wj = w[None, :]
    S  = wi + wj
    R  = wi / np.maximum(S, 1e-300)
    H  = ((R - 0.5)**2 - c_alpha) * (S - Vbar)
    np.fill_diagonal(H, 0.0)
    return H

def compute_Tn(y, Vbar, c_alpha):
    n = len(y)
    return float(_kernel_matrix(y, Vbar, c_alpha).sum()) / (n*(n-1))

# ── JEL pseudo-values and ratio ────────────────────────────────────────────────
def compute_gof_pseudovalues(y):
    y   = np.asarray(y, float)
    n   = len(y)
    fit = qteg_mle(y)
    if fit is None: return None
    ah, bh      = fit['alpha'], fit['beta']
    Vbar        = 2.0 * ah / bh
    c_alpha     = c_alpha_val(ah)
    Tn          = compute_Tn(y, Vbar, c_alpha)
    pv = np.empty(n)
    for i in range(n):
        y_loo = np.delete(y, i)
        fit_i = qteg_mle(y_loo)
        if fit_i is None: al, bl = ah, bh
        else:             al, bl = fit_i['alpha'], fit_i['beta']
        Vl  = 2.0 * al / max(bl, 1e-8)
        cl  = c_alpha_val(al)
        pv[i] = n * Tn - (n-1) * compute_Tn(y_loo, Vl, cl)
    return dict(pv=pv, Tn=Tn, alpha=ah, beta=bh, c_alpha=c_alpha, n=n)

def _solve_lambda(V, max_iter=200, tol=1e-12):
    W = V; n = len(W)
    pw = W[W > 0]; nw = W[W < 0]
    lo = (-1.0/pw.min() + 1e-10) if len(pw) > 0 else -1e8
    hi = (-1.0/nw.max() - 1e-10) if len(nw) > 0 else  1e8
    lam = 0.0
    for _ in range(max_iter):
        denom = 1.0 + lam * W
        if np.any(denom <= 0): lam *= 0.5; continue
        g  = np.sum(W / denom) / n
        dg = -np.sum((W / denom)**2) / n
        if abs(dg) < 1e-14: break
        step = g / dg
        lam  = float(np.clip(lam - step, lo, hi))
        if abs(step) < tol: break
    return lam

def jel_ratio(pv):
    V = np.asarray(pv, float)
    if np.min(V) >= 0 or np.max(V) <= 0: return np.inf
    try:
        lam = _solve_lambda(V)
        return float(2.0 * np.sum(np.log(1.0 + lam * V)))
    except (ValueError, RuntimeError):
        return np.inf

def bcjel_ratio(pv, jel_val):
    """
    Bartlett-corrected JEL statistic.
    b_hat = 0.5 * mean(pv^4) / mean(pv^2)^2 - 1
    ell_BC = ell_JEL / (1 + b_hat / n)
    Reduces finite-sample size inflation while preserving power.
    Reference: DiCiccio, Hall & Romano (1991 Ann. Stat.);
               Hall & La Scala (1990).
    """
    if not np.isfinite(jel_val): return np.inf
    n  = len(pv)
    m2 = np.mean(pv**2)
    m4 = np.mean(pv**4)
    if m2 < 1e-14: return jel_val
    b_hat = 0.5 * m4 / (m2**2) - 1.0
    denom = 1.0 + b_hat / n
    if denom < 0.1: return jel_val   # safety floor
    return jel_val / denom

# ── EDF statistics ─────────────────────────────────────────────────────────────
def edf_statistics(y, alpha, beta):
    """KS, CvM, and AD statistics under fitted QTEG(alpha, beta)."""
    y = np.sort(np.asarray(y, float))
    n = len(y)
    F = np.clip(qteg_cdf(y, alpha, beta), 1e-10, 1.0 - 1e-10)
    i = np.arange(1, n + 1)
    # Kolmogorov-Smirnov
    D_plus  = np.max(i / n - F)
    D_minus = np.max(F - (i - 1) / n)
    ks  = float(max(D_plus, D_minus))
    # Cramer-von Mises
    cvm = float(1.0 / (12 * n) + np.sum((F - (2*i - 1) / (2*n))**2))
    # Anderson-Darling
    ad  = float(-n - np.mean((2*i - 1) * (np.log(F) + np.log(1.0 - F[::-1]))))
    return {"ks": ks, "cvm": cvm, "ad": ad}

# ── Bootstrap calibration for EDF tests ───────────────────────────────────────
def bootstrap_edf_pvalues(y, alpha_hat, beta_hat, B=B_BOOTSTRAP, rng=None):
    """
    Parametric bootstrap p-values for KS, CvM, AD under composite QTEG null.
    Each bootstrap sample is drawn from QTEG(alpha_hat, beta_hat), re-fitted,
    and the EDF statistics are computed under the re-fitted model.
    The p-value is the proportion of bootstrap statistics >= observed statistic.
    """
    if rng is None: rng = np.random.default_rng()
    n   = len(y)
    obs = edf_statistics(y, alpha_hat, beta_hat)
    boot = {"ks": [], "cvm": [], "ad": []}
    for _ in range(B):
        yb  = qteg_sample(alpha_hat, beta_hat, n, rng)
        fb  = qteg_mle(yb)
        if fb is None: continue
        sb  = edf_statistics(yb, fb['alpha'], fb['beta'])
        for k in boot: boot[k].append(sb[k])
    out = {}
    for k in boot:
        v = np.asarray(boot[k])
        m = len(v)
        out[f"{k}_stat"] = obs[k]
        out[f"{k}_pval"] = float((1 + np.sum(v >= obs[k])) / (m + 1)) if m > 0 else np.nan
    out["n_boot_conv"] = len(boot["ks"])
    return out

# ── Full test per replication ──────────────────────────────────────────────────
def run_full_test(y, seed, B=B_BOOTSTRAP):
    """
    Run JEL + bootstrap-calibrated KS/AD/CvM on sample y.
    Returns dict with all results, or None if MLE fails.
    """
    y = np.asarray(y, float)
    y = y[y > 0]
    if len(y) < 5: return None

    # ── JEL ──────────────────────────────────────────────────────────────────
    gof = compute_gof_pseudovalues(y)
    if gof is None: return None
    pv       = gof['pv']
    jel_stat  = jel_ratio(pv)
    jel_pval  = float(chi2.sf(jel_stat, df=1)) if np.isfinite(jel_stat) else 0.0
    bcjel     = bcjel_ratio(pv, jel_stat)
    bcjel_pval = float(chi2.sf(bcjel, df=1)) if np.isfinite(bcjel) else 0.0
    ah, bh   = gof['alpha'], gof['beta']

    # ── EDF bootstrap ─────────────────────────────────────────────────────────
    boot_rng = np.random.default_rng(seed)
    edf = bootstrap_edf_pvalues(y, ah, bh, B=B, rng=boot_rng)

    return dict(
        jel_stat=jel_stat,  jel_pval=jel_pval,
        bcjel_stat=bcjel,   bcjel_pval=bcjel_pval,
        alpha_hat=ah, beta_hat=bh,
        **edf
    )

# ── Monte Carlo standard error ─────────────────────────────────────────────────
def mc_se(rej_count, n_conv):
    p = rej_count / max(n_conv, 1)
    return 100.0 * math.sqrt(p * (1.0 - p) / max(n_conv, 1))

# ── Summary helper ─────────────────────────────────────────────────────────────
def _make_summary(counts, n_sim, meta):
    nc = max(counts['n_conv'], 1)
    out = {**meta,
           'n_sim':    n_sim,
           'n_conv':   counts['n_conv'],
           'rej_jel':   100.0 * counts['rej_jel']   / nc,
           'rej_bcjel': 100.0 * counts['rej_bcjel'] / nc,
           'rej_ks':    100.0 * counts['rej_ks']    / nc,
           'rej_ad':    100.0 * counts['rej_ad']    / nc,
           'rej_cvm':   100.0 * counts['rej_cvm']   / nc,
           'se_jel':    mc_se(counts['rej_jel'],   nc),
           'se_bcjel':  mc_se(counts['rej_bcjel'], nc),
           'se_ks':     mc_se(counts['rej_ks'],    nc),
           'se_ad':     mc_se(counts['rej_ad'],    nc),
           'se_cvm':    mc_se(counts['rej_cvm'],   nc),
           'mean_jel':  counts['sum_jel'] / nc,
           'timestamp': datetime.now().isoformat()}
    return out

def save_partial(block_id, counts, rep, n_sim, meta):
    s = {**_make_summary(counts, n_sim, meta), 'status': 'partial', 'rep_done': rep+1}
    with open(os.path.join(RESULTS_DIR, f"gof_block_{block_id:02d}_partial.json"), 'w') as f:
        json.dump(s, f, indent=2)

def finalise(block_id, counts, n_sim, meta):
    result  = {**_make_summary(counts, n_sim, meta), 'status': 'complete'}
    final   = os.path.join(RESULTS_DIR, f"gof_block_{block_id:02d}.json")
    partial = os.path.join(RESULTS_DIR, f"gof_block_{block_id:02d}_partial.json")
    with open(final, 'w') as f: json.dump(result, f, indent=2)
    if os.path.exists(partial): os.remove(partial)
    return result

# ── Size block ─────────────────────────────────────────────────────────────────
def run_size_block(block_id, n_sim=N_SIM_DEFAULT, B=B_BOOTSTRAP):
    sc_idx          = block_id // len(SAMPLE_SIZES)
    n_idx           = block_id %  len(SAMPLE_SIZES)
    alpha_t, beta_t = NULL_SCENARIOS[sc_idx]
    n               = SAMPLE_SIZES[n_idx]
    seed            = BASE_SEED + block_id * 10000
    meta = dict(block_id=block_id, study='size', scenario=sc_idx+1,
                alpha_t=alpha_t, beta_t=beta_t, n=n, delta=NOMINAL, B=B)

    print(f"\n{'='*65}")
    print(f"[SIZE] Block {block_id:02d} | Sc.{sc_idx+1}(a={alpha_t},b={beta_t})"
          f" | n={n} | n_sim={n_sim} | B={B}")
    print(f"{'='*65}")

    counts    = dict(n_conv=0, rej_jel=0, rej_bcjel=0, rej_ks=0, rej_ad=0, rej_cvm=0, sum_jel=0.0)
    chi2_crit = chi2.ppf(1.0 - NOMINAL, df=1)
    t_start   = time.time()

    for rep in range(n_sim):
        rng  = np.random.default_rng(seed + rep)
        y    = qteg_sample(alpha_t, beta_t, n, rng)
        r    = run_full_test(y, seed=seed + rep + 1000000, B=B)
        if r is None: continue
        counts['n_conv'] += 1
        if np.isfinite(r['jel_stat']) and r['jel_stat'] > chi2_crit:
            counts['rej_jel'] += 1
        if np.isfinite(r.get('bcjel_stat', np.inf)) and r['bcjel_stat'] > chi2_crit:
            counts['rej_bcjel'] += 1
        if r.get('ks_pval',  1.0) < NOMINAL: counts['rej_ks']  += 1
        if r.get('ad_pval',  1.0) < NOMINAL: counts['rej_ad']  += 1
        if r.get('cvm_pval', 1.0) < NOMINAL: counts['rej_cvm'] += 1
        counts['sum_jel'] += r['jel_stat'] if np.isfinite(r['jel_stat']) else 0.0

        if (rep+1) % CHECKPOINT_EVERY == 0:
            nc      = max(counts['n_conv'], 1)
            elapsed = time.time() - t_start
            remain  = (elapsed/(rep+1)) * (n_sim - rep - 1)
            print(f"  Rep {rep+1:>5}/{n_sim}  conv={nc}"
                  f"  JEL={100*counts['rej_jel']/nc:.1f}%"
                  f"  BC-JEL={100*counts['rej_bcjel']/nc:.1f}%"
                  f"  KS={100*counts['rej_ks']/nc:.1f}%"
                  f"  AD={100*counts['rej_ad']/nc:.1f}%"
                  f"  CvM={100*counts['rej_cvm']/nc:.1f}%"
                  f"  elapsed={elapsed:.0f}s  remain~{remain:.0f}s")
            save_partial(block_id, counts, rep, n_sim, meta)

    result = finalise(block_id, counts, n_sim, meta)
    _print_summary(block_id, result, counts, n_sim, t_start)
    return result

# ── Power block ────────────────────────────────────────────────────────────────
def run_power_block(block_id, n_sim=N_SIM_DEFAULT, B=B_BOOTSTRAP):
    pb                     = block_id - N_SIZE_BLOCKS
    sc_idx                 = pb // (len(ALTERNATIVES) * len(SAMPLE_SIZES))
    rem                    = pb %  (len(ALTERNATIVES) * len(SAMPLE_SIZES))
    alt_idx                = rem // len(SAMPLE_SIZES)
    n_idx                  = rem %  len(SAMPLE_SIZES)
    alt_label, alt_sampler = ALTERNATIVES[alt_idx]
    n                      = SAMPLE_SIZES[n_idx]
    null_alpha, null_beta  = NULL_SCENARIOS[sc_idx]
    seed                   = BASE_SEED + block_id * 10000
    meta = dict(block_id=block_id, study='power', alt_label=alt_label,
                n=n, delta=NOMINAL, B=B,
                null_alpha=null_alpha, null_beta=null_beta,
                scenario=sc_idx+1)

    print(f"\n{'='*65}")
    print(f"[POWER] Block {block_id:02d} | Sc.{sc_idx+1}({null_alpha},{null_beta})"
          f" | Alt={alt_label} | n={n} | n_sim={n_sim} | B={B}")
    print(f"{'='*65}")

    counts    = dict(n_conv=0, rej_jel=0, rej_bcjel=0, rej_ks=0, rej_ad=0, rej_cvm=0, sum_jel=0.0)
    chi2_crit = chi2.ppf(1.0 - NOMINAL, df=1)
    t_start   = time.time()

    for rep in range(n_sim):
        rng = np.random.default_rng(seed + rep)
        y   = alt_sampler(n, rng)
        r   = run_full_test(y, seed=seed + rep + 1000000, B=B)
        if r is None: continue
        counts['n_conv'] += 1
        if np.isfinite(r['jel_stat']) and r['jel_stat'] > chi2_crit:
            counts['rej_jel'] += 1
        if np.isfinite(r.get('bcjel_stat', np.inf)) and r['bcjel_stat'] > chi2_crit:
            counts['rej_bcjel'] += 1
        if r.get('ks_pval',  1.0) < NOMINAL: counts['rej_ks']  += 1
        if r.get('ad_pval',  1.0) < NOMINAL: counts['rej_ad']  += 1
        if r.get('cvm_pval', 1.0) < NOMINAL: counts['rej_cvm'] += 1
        counts['sum_jel'] += r['jel_stat'] if np.isfinite(r['jel_stat']) else 0.0

        if (rep+1) % CHECKPOINT_EVERY == 0:
            nc      = max(counts['n_conv'], 1)
            elapsed = time.time() - t_start
            remain  = (elapsed/(rep+1)) * (n_sim - rep - 1)
            print(f"  Rep {rep+1:>5}/{n_sim}  conv={nc}"
                  f"  JEL={100*counts['rej_jel']/nc:.1f}%"
                  f"  BC-JEL={100*counts['rej_bcjel']/nc:.1f}%"
                  f"  KS={100*counts['rej_ks']/nc:.1f}%"
                  f"  AD={100*counts['rej_ad']/nc:.1f}%"
                  f"  CvM={100*counts['rej_cvm']/nc:.1f}%"
                  f"  elapsed={elapsed:.0f}s  remain~{remain:.0f}s")
            save_partial(block_id, counts, rep, n_sim, meta)

    result = finalise(block_id, counts, n_sim, meta)
    _print_summary(block_id, result, counts, n_sim, t_start)
    return result

def _print_summary(block_id, result, counts, n_sim, t_start):
    nc = max(counts['n_conv'], 1)
    print(f"\n--- Block {block_id:02d} complete ---")
    print(f"  Conv:    {counts['n_conv']}/{n_sim}")
    print(f"  JEL:     {result['rej_jel']:.2f}%  (SE={result['se_jel']:.3f}%)")
    print(f"  BC-JEL:  {result['rej_bcjel']:.2f}%  (SE={result['se_bcjel']:.3f}%)")
    print(f"  KS:      {result['rej_ks']:.2f}%  (SE={result['se_ks']:.3f}%)")
    print(f"  AD:      {result['rej_ad']:.2f}%  (SE={result['se_ad']:.3f}%)")
    print(f"  CvM:     {result['rej_cvm']:.2f}%  (SE={result['se_cvm']:.3f}%)")
    print(f"  Time:    {time.time()-t_start:.0f}s")

def run_block(block_id, n_sim=N_SIM_DEFAULT, B=B_BOOTSTRAP):
    if block_id < N_SIZE_BLOCKS:
        return run_size_block(block_id, n_sim, B)
    elif block_id < N_TOTAL_BLOCKS:
        return run_power_block(block_id, n_sim, B)
    else:
        raise ValueError(f"block_id {block_id} out of range [0,{N_TOTAL_BLOCKS-1}]")

# ── Real datasets ──────────────────────────────────────────────────────────────
def _parse(raw):
    return np.array([float(x) for x in re.findall(r'\d+\.?\d*', raw)])

DATASETS = {
    'DS1: Bladder Cancer (n=128)': _parse(
        "0.08,2.09,3.48,4.87,6.94,8.66,13.11,23.63,0.20,2.23,3.52,4.98,"
        "6.97,9.02,13.29,0.40,2.26,3.57,5.06,7.09,9.22,13.80,25.74,0.50,"
        "2.46,3.64,5.09,7.26,9.47,14.24,25.82,0.51,2.54,3.70,5.17,7.28,"
        "9.74,14.76,26.31,0.81,2.62,3.82,5.32,7.32,10.06,14.77,32.15,2.64,"
        "11.79,18.10,1.46,4.40,5.85,8.26,11.98,19.13,1.76,3.25,4.50,6.25,"
        "8.37,12.02,2.02,3.31,4.51,6.54,8.53,12.03,20.28,2.02,3.36,6.76,"
        "12.07,21.73,2.0,3.36,6.93,8.65,12.63,22.69,3.88,5.32,7.39,10.34,"
        "14.83,34.26,0.90,2.69,4.18,5.34,7.59,10.66,15.96,36.66,1.05,2.69,"
        "4.23,5.41,7.62,10.75,16.62,43.01,1.19,2.75,4.26,5.41,7.63,17.12,"
        "46.12,1.26,2.83,4.33,5.49,7.66,11.25,17.14,79.05,1.35,2.87,5.62,"
        "7.87,11.64,17.36,1.40,3.02,4.34,5.71,7.93"),
    'DS2: Boeing 720 (n=213)': _parse(
        "194,413,90,74,55,23,97,50,359,50,130,487,102,15,14,10,57,320,261,"
        "51,44,9,254,493,18,209,41,58,60,48,56,87,11,102,12,5,100,14,29,"
        "37,186,29,104,7,4,72,270,283,7,57,33,100,61,502,220,120,141,22,"
        "603,35,98,54,181,65,49,12,239,14,18,39,3,12,5,32,9,14,70,47,62,"
        "142,3,104,85,67,169,24,21,246,47,68,15,2,91,59,447,56,29,176,225,"
        "77,197,438,43,134,184,20,386,182,71,80,188,230,152,36,79,59,33,"
        "246,1,79,3,27,201,84,27,21,16,88,130,14,118,44,15,42,106,46,230,"
        "59,153,104,20,206,5,66,34,29,26,35,5,82,5,61,31,118,326,12,54,"
        "36,34,18,25,120,31,22,18,156,11,216,139,67,310,3,46,210,57,76,"
        "14,111,97,62,26,71,39,30,7,44,11,63,23,22,23,14,18,13,34,62,11,"
        "191,14,16,18,130,90,163,208,1,24,70,16,101,52,208,95"),
    'DS3: Malignant Melanoma (n=205)': _parse(
        "6.76,0.65,1.34,2.90,12.08,4.84,5.16,3.22,12.88,7.41,4.19,0.16,3.87,"
        "4.84,2.42,12.56,5.80,7.06,5.48,7.73,13.85,2.34,4.19,4.04,4.84,0.32,"
        "8.54,2.58,3.56,3.54,0.97,4.83,1.62,6.44,14.66,2.58,3.87,3.54,1.34,"
        "2.24,3.87,3.54,17.42,1.29,3.22,1.29,4.51,8.38,1.94,0.16,2.58,1.29,"
        "0.16,1.62,1.29,2.10,0.32,0.81,1.13,5.16,1.62,1.37,0.24,0.81,1.29,"
        "1.29,0.97,1.13,5.80,1.29,0.48,1.62,2.26,0.58,0.97,2.58,0.81,3.54,"
        "0.97,1.78,1.94,1.29,3.22,1.53,1.29,1.62,1.62,0.32,4.84,1.29,0.97,"
        "3.06,3.54,1.62,2.58,1.94,0.81,7.73,0.97,12.88,2.58,4.09,0.64,0.97,"
        "3.22,1.62,3.87,0.32,0.32,3.22,2.26,3.06,2.58,0.65,1.13,0.81,0.97,"
        "1.76,1.94,0.65,0.97,5.64,9.66,0.10,5.48,2.26,4.83,0.97,0.97,5.16,"
        "0.81,2.90,3.87,1.94,0.16,0.64,2.26,1.45,4.82,1.29,7.89,0.81,3.54,"
        "1.29,0.64,3.22,1.45,0.48,1.94,0.16,0.16,1.29,1.94,3.54,0.81,0.65,"
        "7.09,0.16,1.62,1.62,1.29,6.12,0.48,0.64,3.22,1.94,2.58,2.58,0.81,"
        "0.81,3.22,0.32,3.22,2.74,4.84,1.62,0.65,1.45,0.65,1.29,1.62,3.54,"
        "3.22,0.65,1.03,7.09,1.29,0.65,1.78,12.24,8.06,0.81,2.10,3.87,0.65,"
        "1.94,0.65,2.10,1.94,1.13,7.06,6.12,0.48,2.26,2.90"),
    'DS4: Guinea Pig Survival (n=72)': _parse(
        "0.1,0.33,0.44,0.56,0.59,0.72,0.74,0.77,0.92,0.93,0.96,1.0,"
        "1.0,1.02,1.05,1.07,1.07,1.08,1.08,1.09,1.12,1.13,1.15,1.16,"
        "1.2,1.21,1.22,1.22,1.24,1.3,1.34,1.36,1.39,1.44,1.46,1.53,"
        "1.59,1.6,1.63,1.68,1.71,1.72,1.76,1.83,1.95,1.96,1.97,2.02,"
        "2.13,2.15,2.16,2.22,2.3,2.31,2.4,2.45,2.51,2.53,2.54,2.54,"
        "2.78,2.93,3.27,3.42,3.47,3.61,4.02,4.32,4.58,5.55,6.0,9.4"),
}

# ── Real data application ──────────────────────────────────────────────────────
def run_real_data(B=B_BOOTSTRAP):
    print("\n" + "="*80)
    print("REAL DATA: JEL + Bootstrap-Calibrated EDF Tests")
    print("="*80)
    all_results = {}
    chi2_crit   = chi2.ppf(0.95, df=1)
    for ds_name, y in DATASETS.items():
        print(f"\n{'─'*60}\n  {ds_name}\n{'─'*60}")
        fit = qteg_mle(y)
        if fit is None: print("MLE FAILED"); continue
        ah, bh = fit['alpha'], fit['beta']
        gof    = compute_gof_pseudovalues(y)
        if gof is None: print("FAILED"); continue
        pv       = gof['pv']
        jel_stat  = jel_ratio(pv)
        jel_pval  = float(chi2.sf(jel_stat, df=1)) if np.isfinite(jel_stat) else 0.0
        bcjel_val = bcjel_ratio(pv, jel_stat)
        bcjel_pval = float(chi2.sf(bcjel_val, df=1)) if np.isfinite(bcjel_val) else 0.0
        rng_rd    = np.random.default_rng(99999)
        edf       = bootstrap_edf_pvalues(y, ah, bh, B=B, rng=rng_rd)
        dec       = lambda p: "REJECT" if p < 0.05 else "do not reject"
        print(f"  alpha={ah:.4f}  beta={bh:.4f}")
        print(f"  JEL:    stat={jel_stat:.4f}   p={jel_pval:.4f}  [{dec(jel_pval)}]")
        print(f"  BC-JEL: stat={bcjel_val:.4f}   p={bcjel_pval:.4f}  [{dec(bcjel_pval)}]")
        print(f"  KS:     stat={edf['ks_stat']:.4f}   p={edf['ks_pval']:.4f}")
        print(f"  AD:     stat={edf['ad_stat']:.4f}   p={edf['ad_pval']:.4f}")
        print(f"  CvM:    stat={edf['cvm_stat']:.4f}   p={edf['cvm_pval']:.4f}")
        all_results[ds_name] = dict(
            n=int(len(y)), alpha_hat=float(ah), beta_hat=float(bh),
            logL=float(fit['logL']), Tn=float(gof['Tn']),
            jel_stat=float(jel_stat),    jel_pval=float(jel_pval),
            bcjel_stat=float(bcjel_val), bcjel_pval=float(bcjel_pval),
            ks_stat=edf['ks_stat'],      ks_pval=edf['ks_pval'],
            ad_stat=edf['ad_stat'],      ad_pval=edf['ad_pval'],
            cvm_stat=edf['cvm_stat'],    cvm_pval=edf['cvm_pval'],
            n_boot_conv=edf['n_boot_conv'],
            decision_jel=dec(jel_pval), pv=pv.tolist(), y=y.tolist()
        )
    out = os.path.join(RESULTS_DIR, 'QTEG_GoF_realdata_results.json')
    with open(out, 'w') as f: json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")
    return all_results

# ── Merge ──────────────────────────────────────────────────────────────────────
def merge_results():
    results, missing = [], []
    for i in range(N_TOTAL_BLOCKS):
        fname = os.path.join(RESULTS_DIR, f"gof_block_{i:02d}.json")
        if os.path.exists(fname):
            with open(fname) as f: r = json.load(f)
            results.append(r)
        else:
            missing.append(i)
    if missing: print(f"Missing blocks: {missing}")
    else: print(f"All {N_TOTAL_BLOCKS} blocks complete.")
    json_out = os.path.join(RESULTS_DIR, 'QTEG_GoF_full_results.json')
    with open(json_out, 'w') as f: json.dump(results, f, indent=2)
    if results:
        csv_out = os.path.join(RESULTS_DIR, 'QTEG_GoF_full_results.csv')
        # Collect ALL keys across all blocks so no column is ever missing
        all_keys = []
        seen = set()
        for r in results:
            for k in r.keys():
                if k not in seen:
                    all_keys.append(k); seen.add(k)
        with open(csv_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore',
                               restval='')
            w.writeheader(); w.writerows(results)
        print(f"CSV -> {csv_out}")
    # Console summary
    size_rows  = [r for r in results if r.get('study')=='size']
    power_rows = [r for r in results if r.get('study')=='power']
    if size_rows:
        print(f"\n{'Sc':<4} {'n':>4}  {'JEL%':>7} {'BC-JEL%':>8} {'KS%':>7} {'AD%':>7} {'CvM%':>7}")
        print("-"*50)
        for r in sorted(size_rows, key=lambda x:(x['scenario'],x['n'])):
            print(f"Sc.{r['scenario']} {r['n']:>4}  "
                  f"{r['rej_jel']:>7.2f} {r.get('rej_bcjel',0.):>8.2f} "
                  f"{r['rej_ks']:>7.2f} {r['rej_ad']:>7.2f} {r['rej_cvm']:>7.2f}")
    if power_rows:
        for sc_idx, (na, nb) in enumerate(NULL_SCENARIOS):
            sc_rows = [r for r in power_rows if r.get('null_alpha')==na and r.get('null_beta')==nb]
            if not sc_rows: continue
            print(f"\nPower -- Sc.{sc_idx+1}: QTEG({na},{nb})")
            print(f"{'Alternative':<22} {'n':>4}  {'JEL%':>7} {'BC-JEL%':>8} {'KS%':>7} {'AD%':>7} {'CvM%':>7}")
            print("-"*65)
            for r in sorted(sc_rows, key=lambda x:(x['alt_label'],x['n'])):
                print(f"{r['alt_label']:<22} {r['n']:>4}  "
                      f"{r['rej_jel']:>7.2f} {r.get('rej_bcjel',0.):>8.2f} "
                      f"{r['rej_ks']:>7.2f} {r['rej_ad']:>7.2f} {r['rej_cvm']:>7.2f}")

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QTEG GoF v4 -- JEL + EDF')
    parser.add_argument('--block',    type=int, default=None)
    parser.add_argument('--n_sim',    type=int, default=N_SIM_DEFAULT)
    parser.add_argument('--B',        type=int, default=B_BOOTSTRAP)
    parser.add_argument('--merge',    action='store_true')
    parser.add_argument('--realdata', action='store_true')
    parser.add_argument('--test',     action='store_true')
    args = parser.parse_args()
    if args.test:
        print("TEST: block 0, 3 reps, B=10")
        run_block(0, n_sim=3, B=10)
    elif args.realdata:
        run_real_data(B=args.B)
    elif args.merge:
        merge_results()
    elif args.block is not None:
        run_block(args.block, n_sim=args.n_sim, B=args.B)
    else:
        print(f"QTEG GoF v4: {N_TOTAL_BLOCKS} blocks | n_sim={N_SIM_DEFAULT} | B={B_BOOTSTRAP}")
        print("  Usage:")
        print("    python QTEG_GoF_Arctic_v4.py --test")
        print("    sbatch qteg_gof_array_v4.sh")
        print("    python QTEG_GoF_Arctic_v4.py --merge")
        print("    python QTEG_GoF_Arctic_v4.py --realdata")
