"""
================================================================================
QTEG_GoF_plot_results_v4.py  --  JEL + KS + AD + CvM
================================================================================
Generates all paper figures from unified v4 simulation results.
Requires QTEG_GoF_full_results.csv and QTEG_GoF_realdata_results.json
produced by QTEG_GoF_Arctic_v4.py --merge and --realdata.

Figure layout:
  Figure 1  fig1_size_all.png/pdf
            1x3: empirical size for all 4 tests across 3 null scenarios
            -- shows all tests properly calibrated at 5%

  Figure 2  fig2_power_main.png/pdf
            2x3: power for 3 main alternatives (Wb1.5, LogN, Exp)
            top row n=50..200, bottom row n=50..200 -- 4 lines per panel

  Figure 3  fig3_power_hard.png/pdf
            1x2: power for 2 hard alternatives (Gam, Wb0.8) -- 4 lines

  Figure 4  fig4_power_summary.png/pdf
            Power at n=200 for all 5 alternatives x 4 tests
            Grouped bar chart -- quick visual comparison

  Figure 5  fig5_sqrtY_overlay.png/pdf
            2x2: KDE vs fitted Gamma density (real data, unchanged)

  Figure 6  fig6_pp_plots.png/pdf
            2x2: PP plots (real data, unchanged)

  Supplementary
            fig_supp_null_calibration.png/pdf
            Mean JEL stat under H0 converging to chi2_1 mean

Run:
  python QTEG_GoF_plot_results_v4.py

Authors: Taiwo Michael Ayeni and Yichuan Zhao, GSU 2026
================================================================================
"""

import os, json, math
import numpy as np
import pandas as pd
from scipy.special import gammaln, gammainc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_FILE    = os.path.join(RESULTS_DIR, "QTEG_GoF_full_results.csv")
RD_FILE     = os.path.join(RESULTS_DIR, "QTEG_GoF_realdata_results.json")

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Not found: {CSV_FILE}\nRun --merge first.")

df = pd.read_csv(CSV_FILE)

# Fill missing scenario values for old Sc.1 power blocks (scenario=NaN, null_alpha=1.5)
if "scenario" in df.columns and "null_alpha" in df.columns:
    alpha_to_sc = {1.5: 1.0, 2.0: 2.0, 3.0: 3.0}
    mask = (df["study"] == "power") & df["scenario"].isna() & df["null_alpha"].notna()
    df.loc[mask, "scenario"] = df.loc[mask, "null_alpha"].map(alpha_to_sc)

NS = sorted(df["n"].unique().tolist())

# Check which test columns are available
HAS_EDF = all(c in df.columns for c in ["rej_ks","rej_ad","rej_cvm"])
if not HAS_EDF:
    print("WARNING: rej_ks/rej_ad/rej_cvm columns not found.")
    print("         Figures will show JEL only.")

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "#cccccc",
    "axes.grid":         True,
    "grid.alpha":        0.22,
    "grid.linestyle":    ":",
    "grid.linewidth":    0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "lines.linewidth":   1.8,
    "lines.markersize":  5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})

# ── Design tokens ──────────────────────────────────────────────────────────────
TESTS = {
    "jel":   dict(col="#1a6b3a", ls="-",             mk="o", lw=2.0, label="JEL"),
    "bcjel": dict(col="#0e7c7b", ls=(0,(4,1,1,1)),   mk="D", lw=1.8, label="BC-JEL"),
    "ks":    dict(col="#1f4e79", ls="--",             mk="s", lw=1.6, label="KS"),
    "ad":    dict(col="#8b1a1a", ls=":",              mk="^", lw=1.6, label="AD"),
    "cvm":   dict(col="#7b4e00", ls="-.",             mk="P", lw=1.6, label="CvM"),
}
# Auto-detect which tests are in the CSV
ACTIVE_TESTS = []
for t in ["jel","bcjel","ks","ad","cvm"]:
    if f"rej_{t}" in df.columns:
        ACTIVE_TESTS.append(t)

NOM_COLOR = "#555555"
BAND_COLOR = "#bbbbbb"

SC_SHORT = {
    1: r"QTEG($\alpha=1.5,\;\beta=0.5$)",
    2: r"QTEG($\alpha=2.0,\;\beta=1.0$)",
    3: r"QTEG($\alpha=3.0,\;\beta=2.0$)",
}
ALT_LABELS = {
    "Weibull(0.8)":   r"Weibull($\kappa\!=\!0.8$)",
    "Weibull(1.5)":   r"Weibull($\kappa\!=\!1.5$)",
    "LogNormal(0,1)": r"Log-Normal$(0,1)$",
    "Exponential(1)": r"Exponential$(1)$",
    "Gamma(2,1)":     r"Gamma$(2,1)$",
}
NULL_SUBTITLE = r"$H_0: Y \sim \mathrm{QTEG}(1.5,\;0.5)$; $N=5{,}000$; $\delta=5\%$"

# ── Helpers ────────────────────────────────────────────────────────────────────
def savefig(fig, stem):
    for ext in [".pdf", ".png"]:
        p = os.path.join(RESULTS_DIR, stem + ext)
        fig.savefig(p)
        print(f"  Saved: {p}")
    plt.close(fig)

def _get_jel_sizes():
    size_df = df[df["study"] == "size"].copy()
    return size_df.groupby("n")["rej_jel"].mean().to_dict()

def _get_bcjel_sizes():
    size_df = df[df["study"] == "size"].copy()
    if "rej_bcjel" not in size_df.columns:
        return {}
    return size_df.groupby("n")["rej_bcjel"].mean().to_dict()

JEL_SIZES   = _get_jel_sizes()
BCJEL_SIZES = _get_bcjel_sizes()

def _set_xticks(ax, ns):
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlim(ns[0]-8, ns[-1]+8)

def _add_jel_size_reference(ax, ns):
    """
    Dotted line showing JEL empirical size under QTEG null as a function of n.
    Standard Option A reference on power figures -- shows power vs size gap.
    """
    size_vals = [JEL_SIZES.get(n, None) for n in ns]
    if any(v is not None for v in size_vals):
        ax.plot(ns, size_vals,
                color=TESTS["jel"]["col"], linestyle=":",
                linewidth=1.4, marker="", zorder=3, alpha=0.70,
                label=r"JEL size (QTEG null)")

def _clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _plot_test_lines(ax, ns, sub, tests, se_band=False):
    """Plot one line per test on ax from sub dataframe."""
    for t in tests:
        tk = TESTS[t]
        col_r = f"rej_{t}"; col_s = f"se_{t}"
        if col_r not in sub.columns: continue
        rej = sub.sort_values("n")[col_r].tolist()
        ax.plot(ns, rej, color=tk["col"], linestyle=tk["ls"],
                marker=tk["mk"], linewidth=tk["lw"],
                markersize=5, zorder=4, label=tk["label"])
        if se_band and col_s in sub.columns:
            se = sub.sort_values("n")[col_s].tolist()
            ax.fill_between(ns,
                [r-s for r,s in zip(rej,se)],
                [r+s for r,s in zip(rej,se)],
                color=tk["col"], alpha=0.10, linewidth=0)

def _legend_handles(tests, include_size_ref=False):
    handles = []
    for t in tests:
        tk = TESTS[t]
        handles.append(Line2D([0],[0], color=tk["col"], linestyle=tk["ls"],
                               marker=tk["mk"], linewidth=tk["lw"],
                               markersize=5, label=tk["label"]))
    handles.append(Line2D([0],[0], color=NOM_COLOR, linestyle="--",
                           linewidth=1.0, label="Nominal 5%"))
    if include_size_ref:
        handles.append(Line2D([0],[0], color=TESTS["jel"]["col"],
                               linestyle=":", linewidth=1.4, alpha=0.70,
                               label=r"JEL size (QTEG null)"))
    return handles

# ── Figure 1: Empirical Size -- all 4 tests ───────────────────────────────────
def plot_figure1_size():
    """
    1x3 panels: empirical size (%) vs n for all tests under 3 null scenarios.
    All tests should converge to 5% -- confirms correct bootstrap calibration.
    """
    size_df   = df[df["study"]=="size"].copy()
    scenarios = sorted(size_df["scenario"].unique())
    has_se    = "se_jel" in size_df.columns

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.9), sharey=True)

    for col, sc in enumerate(scenarios):
        ax  = axes[col]
        sub = size_df[size_df["scenario"]==sc]
        ns  = sorted(sub["n"].unique())

        ax.axhline(5.0, color=NOM_COLOR, ls="--", lw=1.0, zorder=3)
        ax.axhspan(4.0, 6.0, color=BAND_COLOR, alpha=0.18, zorder=1)

        _plot_test_lines(ax, ns, sub, ACTIVE_TESTS, se_band=(col==0))
        ax.set_title(SC_SHORT[sc], fontsize=8.5, pad=5)
        ax.set_xlabel("Sample size $n$", labelpad=4)
        ax.set_ylim(0, 15)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(3))
        _set_xticks(ax, ns)
        _clean_ax(ax)

    axes[0].set_ylabel("Empirical size (%)", labelpad=4)

    handles = _legend_handles(ACTIVE_TESTS)
    handles.append(mpatches.Patch(facecolor=BAND_COLOR, alpha=0.55,
                                   edgecolor="none", label="[4%, 6%] band"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.12), frameon=True, fontsize=8)

    # Figure-level subtitle identifying the null
    fig.text(0.5, 0.99,
             r"$H_0: Y \sim \mathrm{QTEG}(\alpha,\beta)$; "
             r"$N=5{,}000$ replications; $\delta=5\%$",
             ha="center", va="top", fontsize=8.5, style="italic")

    fig.subplots_adjust(left=0.10, right=0.97, top=0.84,
                        bottom=0.26, wspace=0.08)
    savefig(fig, "fig1_size_all")


# ── Figure 2: Power -- 3 main alternatives (all 5 tests, Sc.1) ───────────────
SC_COLORS = ["#1a6b3a", "#1f4e79", "#8b1a1a"]
SC_LSTYLE = ["-", "--", ":"]
SC_MARKER = ["o", "s", "^"]

def plot_figure2_power_main():
    """
    1x3 panels: Weibull(1.5), LogNormal(0,1), Exponential(1).
    All 5 tests shown, Sc.1 null only. Same design as original.
    """
    power_df  = df[df["study"]=="power"].copy()
    # Use Sc.1 only
    if "scenario" in power_df.columns:
        power_df = power_df[power_df["scenario"]==1.0]
    elif "null_alpha" in power_df.columns:
        power_df = power_df[power_df["null_alpha"]==1.5]
    main_alts = ["Weibull(1.5)", "LogNormal(0,1)", "Exponential(1)"]

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.9), sharey=True)

    for col, alt in enumerate(main_alts):
        ax  = axes[col]
        sub = power_df[power_df["alt_label"]==alt]
        if sub.empty: ax.set_visible(False); continue
        ns  = sorted(sub["n"].unique())

        ax.axhline(5.0, color=NOM_COLOR, ls="--", lw=1.0, zorder=3)
        _add_jel_size_reference(ax, ns)
        _plot_test_lines(ax, ns, sub, ACTIVE_TESTS)
        ax.set_title(ALT_LABELS.get(alt, alt), fontsize=8.5, pad=5)
        ax.set_xlabel("Sample size $n$", labelpad=4)
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        _set_xticks(ax, ns)
        _clean_ax(ax)

    axes[0].set_ylabel("Empirical power (%)", labelpad=4)

    fig.legend(handles=_legend_handles(ACTIVE_TESTS, include_size_ref=True),
               loc="lower center", ncol=len(ACTIVE_TESTS)+2,
               bbox_to_anchor=(0.5, -0.12), frameon=True, fontsize=8)

    fig.text(0.5, 0.99, NULL_SUBTITLE,
             ha="center", va="top", fontsize=8.5, style="italic")

    fig.subplots_adjust(left=0.10, right=0.97, top=0.84,
                        bottom=0.26, wspace=0.08)
    savefig(fig, "fig2_power_main")


# ── Figure 3: Power -- 2 hard alternatives (all 5 tests, Sc.1) ───────────────
def plot_figure3_power_hard():
    """
    1x2 panels: Gamma(2,1) and Weibull(0.8) -- near-null alternatives.
    All 5 tests shown, Sc.1 null only. Same design as original.
    """
    power_df  = df[df["study"]=="power"].copy()
    # Use Sc.1 only
    if "scenario" in power_df.columns:
        power_df = power_df[power_df["scenario"]==1.0]
    elif "null_alpha" in power_df.columns:
        power_df = power_df[power_df["null_alpha"]==1.5]
    hard_alts = ["Gamma(2,1)", "Weibull(0.8)"]

    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.9), sharey=True)

    for col, alt in enumerate(hard_alts):
        ax  = axes[col]
        sub = power_df[power_df["alt_label"]==alt]
        if sub.empty: ax.set_visible(False); continue
        ns  = sorted(sub["n"].unique())

        ax.axhline(5.0, color=NOM_COLOR, ls="--", lw=1.0, zorder=3)
        _add_jel_size_reference(ax, ns)
        _plot_test_lines(ax, ns, sub, ACTIVE_TESTS)
        ax.set_title(ALT_LABELS.get(alt, alt), fontsize=8.5, pad=5)
        ax.set_xlabel("Sample size $n$", labelpad=4)
        ax.set_ylim(0, 70)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
        _set_xticks(ax, ns)
        _clean_ax(ax)

    axes[0].set_ylabel("Empirical power (%)", labelpad=4)

    fig.legend(handles=_legend_handles(ACTIVE_TESTS, include_size_ref=True),
               loc="lower center", ncol=len(ACTIVE_TESTS)+2,
               bbox_to_anchor=(0.5, -0.14), frameon=True, fontsize=8)

    fig.text(0.5, 0.99, NULL_SUBTITLE,
             ha="center", va="top", fontsize=8.5, style="italic")

    fig.subplots_adjust(left=0.13, right=0.97, top=0.84,
                        bottom=0.30, wspace=0.08)
    savefig(fig, "fig3_power_hard")


# ── Figure 4: Power summary bar chart at n=200 ────────────────────────────────
def plot_figure4_power_summary():
    """
    Grouped bar chart: power at n=200 for all 5 alternatives x 4 tests.
    This is the key "at a glance" comparison figure requested by reviewer.
    Bars grouped by alternative; one colour per test.
    """
    if not HAS_EDF:
        print("  Skipping Figure 4 (no EDF columns).")
        return

    power_df = df[(df["study"]=="power") & (df["n"]==200)].copy()
    # Use Sc.1 for bar chart (representative, keeps figure readable)
    if "scenario" in power_df.columns and not power_df["scenario"].isna().all():
        power_df = power_df[power_df["scenario"]==1.0]
    elif "null_alpha" in power_df.columns:
        power_df = power_df[power_df["null_alpha"]==1.5]
    all_alts = ["Weibull(1.5)", "LogNormal(0,1)", "Exponential(1)",
                "Gamma(2,1)", "Weibull(0.8)"]
    all_alts = [a for a in all_alts if a in power_df["alt_label"].values]

    x    = np.arange(len(all_alts))
    ntests = len(ACTIVE_TESTS)
    width  = 0.18
    offsets = np.linspace(-(ntests-1)/2, (ntests-1)/2, ntests) * width

    fig, ax = plt.subplots(figsize=(6.8, 3.2))

    for j, t in enumerate(ACTIVE_TESTS):
        tk   = TESTS[t]
        vals = []
        ses  = []
        for alt in all_alts:
            row = power_df[power_df["alt_label"]==alt]
            if row.empty:
                vals.append(0); ses.append(0)
            else:
                vals.append(float(row[f"rej_{t}"].iloc[0]))
                se_col = f"se_{t}"
                ses.append(float(row[se_col].iloc[0]) if se_col in row.columns else 0)
        bars = ax.bar(x + offsets[j], vals, width,
                      color=tk["col"], label=tk["label"],
                      alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.errorbar(x + offsets[j], vals, yerr=ses,
                    fmt="none", color="black", capsize=2.5,
                    capthick=0.8, elinewidth=0.8, zorder=5)

    ax.axhline(5.0, color=NOM_COLOR, ls="--", lw=1.0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([ALT_LABELS.get(a, a) for a in all_alts],
                        fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel("Empirical power at $n=200$ (%)", labelpad=4)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    _clean_ax(ax)

    # Legend outside below the figure — not inside the plot
    handles_bar = []
    for t in ACTIVE_TESTS:
        tk = TESTS[t]
        handles_bar.append(mpatches.Patch(facecolor=tk["col"],
                                           alpha=0.85, label=tk["label"]))
    handles_bar.append(Line2D([0],[0], color=NOM_COLOR, ls="--",
                               lw=1.0, label="Nominal 5%"))
    fig.legend(handles=handles_bar, loc="lower center",
               ncol=len(handles_bar),
               bbox_to_anchor=(0.5, -0.10), frameon=True, fontsize=8)

    fig.text(0.5, 0.99, NULL_SUBTITLE,
             ha="center", va="top", fontsize=8.5, style="italic")

    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.28)
    savefig(fig, "fig4_power_summary_n200")


# ── Figure 5: sqrt(Y) KDE overlay (real data) ─────────────────────────────────
def _gamma_pdf(w, alpha, beta):
    lw = np.log(np.maximum(w, 1e-300))
    return np.exp(alpha*np.log(beta) - gammaln(alpha)
                  + (alpha-1)*lw - beta*w)

def plot_figure5_sqrtY():
    if not os.path.exists(RD_FILE):
        print(f"  Skipping Figure 5 -- {RD_FILE} not found."); return
    with open(RD_FILE) as f: results = json.load(f)

    DS_TITLE = {
        "DS1: Bladder Cancer (n=128)":     r"DS1: Bladder Cancer ($n=128$)",
        "DS2: Boeing 720 (n=213)":         r"DS2: Boeing 720 ($n=213$)",
        "DS3: Malignant Melanoma (n=205)": r"DS3: Malignant Melanoma ($n=205$)",
        "DS4: Guinea Pig Survival (n=72)": r"DS4: Guinea Pig ($n=72$)",
    }
    DS_XLABEL = {
        "DS1: Bladder Cancer (n=128)":     r"$\sqrt{Y}$ (months$^{1/2}$)",
        "DS2: Boeing 720 (n=213)":         r"$\sqrt{Y}$ (hours$^{1/2}$)",
        "DS3: Malignant Melanoma (n=205)": r"$\sqrt{Y}$ (years$^{1/2}$)",
        "DS4: Guinea Pig Survival (n=72)": r"$\sqrt{Y}$ (years$^{1/2}$)",
    }
    KDE_COL = "#1f4e79"; FIT_COL = "#c0392b"

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.4))
    for ax, ds_name in zip(axes.flat, list(results.keys())):
        r         = results[ds_name]
        y         = np.array(r["y"]); w = np.sqrt(y)
        ah        = r["alpha_hat"]; bh = r["beta_hat"]
        jel_stat  = r["jel_stat"];  jel_pval = r["jel_pval"]
        n         = r["n"]

        bw     = 1.06 * np.std(w, ddof=1) * n**(-0.2)
        w_grid = np.linspace(max(w.min()*0.6, 1e-4), w.max()*1.08, 400)
        kde    = np.mean(np.exp(-0.5*((w_grid[:,None]-w[None,:])/bw)**2)
                         / (bw*np.sqrt(2*np.pi)), axis=1)
        fitted = _gamma_pdf(w_grid, ah, bh)

        ax.plot(w_grid, kde,    color=KDE_COL, ls="-",  lw=1.8,
                label=r"KDE of $\sqrt{Y_i}$", zorder=4)
        ax.plot(w_grid, fitted, color=FIT_COL, ls="--", lw=1.8,
                label=r"Fitted $\Gamma(\hat{\alpha},\hat{\beta})$", zorder=3)

        ax.text(0.97, 0.96,
                f"$\\ell_{{\\mathrm{{JEL}}}}(0)={jel_stat:.3f}$\n"
                f"$p={jel_pval:.3f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.88, edgecolor="#aaaaaa", lw=0.6))

        ax.set_title(DS_TITLE.get(ds_name, ds_name), fontsize=8.5, pad=5)
        ax.set_xlabel(DS_XLABEL.get(ds_name, r"$\sqrt{Y}$"), labelpad=4)
        ax.set_ylabel("Density", labelpad=4)
        ax.set_xlim(left=0)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        _clean_ax(ax)

    fig.legend(
        handles=[
            Line2D([0],[0], color=KDE_COL, ls="-",  lw=1.8,
                   label=r"KDE of $\sqrt{Y_i}$"),
            Line2D([0],[0], color=FIT_COL, ls="--", lw=1.8,
                   label=r"Fitted QTEG: $\Gamma(\hat{\alpha},\hat{\beta})$"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5,-0.03),
        frameon=True, fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.95,
                        bottom=0.11, hspace=0.50, wspace=0.28)
    savefig(fig, "fig5_sqrtY_overlay")


# ── Figure 6: PP plots (real data) ────────────────────────────────────────────
def _qteg_cdf_pp(y, alpha, beta):
    return gammainc(alpha, beta*np.sqrt(np.maximum(y, 0.0)))

def plot_figure6_pp_plots():
    if not os.path.exists(RD_FILE):
        print(f"  Skipping Figure 6 -- {RD_FILE} not found."); return
    with open(RD_FILE) as f: results = json.load(f)

    DS_TITLE = {
        "DS1: Bladder Cancer (n=128)":     r"DS1: Bladder Cancer ($n=128$)",
        "DS2: Boeing 720 (n=213)":         r"DS2: Boeing 720 ($n=213$)",
        "DS3: Malignant Melanoma (n=205)": r"DS3: Malignant Melanoma ($n=205$)",
        "DS4: Guinea Pig Survival (n=72)": r"DS4: Guinea Pig ($n=72$)",
    }
    DOT_COL = "#1a6b3a"; REF_COL = "#333333"; CI_COL = "#aaaaaa"

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.4))
    for ax, ds_name in zip(axes.flat, list(results.keys())):
        r         = results[ds_name]
        y         = np.sort(np.array(r["y"]))
        n         = len(y)
        ah        = r["alpha_hat"]; bh = r["beta_hat"]
        jel_pval  = r["jel_pval"]

        emp     = (np.arange(1, n+1) - 0.5) / n
        theo    = _qteg_cdf_pp(y, ah, bh)
        max_dev = float(np.max(np.abs(emp - theo)))
        ks_band = 1.36 / np.sqrt(n)
        t_grid  = np.linspace(0, 1, 200)

        ax.plot([0,1],[0,1], color=REF_COL, ls="-", lw=1.0, zorder=2)
        ax.fill_between(t_grid,
                        np.clip(t_grid-ks_band,0,1),
                        np.clip(t_grid+ks_band,0,1),
                        color=CI_COL, alpha=0.20, lw=0, zorder=1)
        ax.scatter(emp, theo, color=DOT_COL, s=12, alpha=0.75, lw=0, zorder=4)

        ax.text(0.04, 0.96,
                f"$\\max|F_n-\\hat{{F}}_{{\\mathrm{{QTEG}}}}|={max_dev:.3f}$\n"
                f"JEL $p={jel_pval:.3f}$",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.88, edgecolor="#aaaaaa", lw=0.6))

        ax.set_title(DS_TITLE.get(ds_name, ds_name), fontsize=8.5, pad=5)
        ax.set_xlabel(r"Empirical CDF $F_n(y)$", labelpad=4)
        ax.set_ylabel(r"Theoretical CDF $\hat{F}(y)$", labelpad=4)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.set_aspect("equal", adjustable="box")
        _clean_ax(ax)

    fig.legend(
        handles=[
            Line2D([0],[0], color=REF_COL, ls="-", lw=1.0,
                   label="Reference line ($y=x$)"),
            mpatches.Patch(facecolor=CI_COL, alpha=0.45, edgecolor="none",
                           label="95% KS band"),
            Line2D([0],[0], color=DOT_COL, ls="none", marker="o",
                   markersize=4, alpha=0.85,
                   label=r"$(F_n(y_{(i)}),\hat{F}(y_{(i)}))$"),
        ],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5,-0.03),
        frameon=True, fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.95,
                        bottom=0.12, hspace=0.52, wspace=0.32)
    savefig(fig, "fig6_pp_plots")


# ── Supplementary: Null calibration ───────────────────────────────────────────
def plot_supp_null_calibration():
    size_df   = df[df["study"]=="size"].copy()
    scenarios = sorted(size_df["scenario"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.6), sharey=True)

    for col, sc in enumerate(scenarios):
        ax  = axes[col]
        sub = size_df[size_df["scenario"]==sc].sort_values("n")
        ns  = sub["n"].tolist()
        ml  = sub["mean_jel"].tolist()
        ax.plot(ns, ml, color=TESTS["jel"]["col"], ls="-",
                marker="o", lw=2.0, ms=5, zorder=4, label="JEL")
        ax.axhline(1.0, color=NOM_COLOR, ls="--", lw=1.0, zorder=3)
        ax.set_title(SC_SHORT[sc], fontsize=8.5, pad=5)
        ax.set_xlabel("Sample size $n$", labelpad=4)
        _set_xticks(ax, ns)
        _clean_ax(ax)

    axes[0].set_ylabel(r"Mean $\ell_{\mathrm{JEL}}(0)$ under $H_0$", labelpad=4)

    fig.text(0.5, 0.99,
             r"$H_0: Y \sim \mathrm{QTEG}(\alpha,\beta)$; $N=5{,}000$ replications",
             ha="center", va="top", fontsize=8.5, style="italic")

    fig.legend(
        handles=[
            Line2D([0],[0], color=TESTS["jel"]["col"], ls="-", marker="o",
                   lw=2.0, ms=5, label=r"Mean $\ell_{\mathrm{JEL}}(0)$"),
            Line2D([0],[0], color=NOM_COLOR, ls="--", lw=1.0,
                   label=r"$\chi^2_1$ mean $= 1$"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5,-0.06),
        frameon=True, fontsize=8)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.84,
                        bottom=0.25, wspace=0.10)
    savefig(fig, "fig_supp_null_calibration")


# ── Console tables ─────────────────────────────────────────────────────────────
def print_tables():
    size_df  = df[df["study"]=="size"].copy()
    power_df = df[df["study"]=="power"].copy()

    print("\n" + "="*75)
    print("TABLE 1 -- Empirical Size (%) | N=5,000 | delta=5%")
    print("="*75)
    scenarios = sorted(size_df["scenario"].unique())
    tests = ACTIVE_TESTS
    hdr = f"{'n':>4}  Sc."
    for sc in scenarios:
        for t in tests:
            hdr += f"  {TESTS[t]['label']:>6}"
    print(hdr)
    print("-"*75)
    for n in sorted(size_df["n"].unique()):
        row = f"{int(n):>4}  "
        for sc in scenarios:
            row += "   "
            for t in tests:
                s = size_df[(size_df["scenario"]==sc)&(size_df["n"]==n)]
                col = f"rej_{t}"
                v = float(s[col].iloc[0]) if not s.empty and col in s.columns else np.nan
                row += f"{v:>6.2f}" if not np.isnan(v) else f"{'--':>6}"
        print(row)

    print("\n" + "="*75)
    print("TABLE 2 -- Empirical Power (%) | N=5,000 | null: QTEG(1.5,0.5)")
    print("="*75)
    all_alts = sorted(power_df["alt_label"].unique())
    hdr2 = f"{'Alternative':<22} {'n':>4}"
    for t in tests: hdr2 += f"  {TESTS[t]['label']:>7}"
    print(hdr2); print("-"*60)
    for alt in all_alts:
        for n in sorted(power_df["n"].unique()):
            s = power_df[(power_df["alt_label"]==alt)&(power_df["n"]==n)]
            if s.empty: continue
            row = f"{alt:<22} {int(n):>4}"
            for t in tests:
                col = f"rej_{t}"
                v = float(s[col].iloc[0]) if col in s.columns else np.nan
                row += f"  {v:>7.2f}" if not np.isnan(v) else f"  {'--':>7}"
            print(row)
        print()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*65)
    print("QTEG GoF v4 -- Generating journal figures (JEL + EDF)")
    print(f"Tests available: {ACTIVE_TESTS}")
    print("="*65)

    print_tables()

    print("\nFigure 1: Empirical size -- all tests ...")
    plot_figure1_size()

    print("Figure 2: Power -- 3 main alternatives ...")
    plot_figure2_power_main()

    print("Figure 3: Power -- 2 hard alternatives ...")
    plot_figure3_power_hard()

    print("Figure 4: Power summary bar chart at n=200 ...")
    plot_figure4_power_summary()

    print("Figure 5: sqrt(Y) KDE overlay -- real data ...")
    plot_figure5_sqrtY()

    print("Figure 6: PP plots -- real data ...")
    plot_figure6_pp_plots()

    print("Supplementary: Null calibration ...")
    plot_supp_null_calibration()

    print(f"\nAll figures saved to: {RESULTS_DIR}")
    print("  fig1_size_all              -- size: JEL + KS + AD + CvM")
    print("  fig2_power_main            -- power: Wb1.5, LogN, Exp")
    print("  fig3_power_hard            -- power: Gam, Wb0.8 (near-null)")
    print("  fig4_power_summary_n200    -- grouped bar: power at n=200")
    print("  fig5_sqrtY_overlay         -- KDE vs fitted Gamma (real data)")
    print("  fig6_pp_plots              -- PP plots (real data)")
    print("  fig_supp_null_calibration  -- Wilks convergence")
