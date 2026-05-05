"""
07_LMM_plots.py
===============

Pure-Python plotting script for Figures 5 and 6 of the LMM analysis.

Reads:
    ../data/predictions_ALSPAC_20_MRI_I.csv      (raw BrainPAD wave I)
    ../data/predictions_ALSPAC_30_MRI_II.csv     (raw BrainPAD wave II)
    ../data/lmm_input_long.csv                   (bias-corrected + YJ BrainPAD)
    ../data/lmm_input_deltas.csv                 (one row per ID with ΔBrainPAD)
    ../data/lmm_results_main.csv                 (q-values from the R LMM)

Writes:
    ../figures/Figure5_raincloud_waveI.svg
    ../figures/Figure5_raincloud_waveII.svg
    ../figures/Figure6_delta_raincloud_LPEs1.svg
    ../figures/Figure6_delta_raincloud_LPEs2.svg

Figure 4 (longitudinal trajectories) is produced by 06_figure3_predictions.py
- run that script directly to generate it.

Run from src/:
    python 07_LMM_plots.py
"""

from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

DATA_DIR = "../data"
FIG_DIR  = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)

PAD_COL = "BrainPAD_c"


# ---------------------------------------------------------------------------
# Helpers (ported from rain_cloud_plot_predictions_ALSPAC, source line 242)
# ---------------------------------------------------------------------------
def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(c) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))


def _interp(start_hex: str, end_hex: str, n: int) -> list[str]:
    a, b = np.array(_hex_to_rgb(start_hex)), np.array(_hex_to_rgb(end_hex))
    return [_rgb_to_hex(a + (b - a) * t) for t in np.linspace(0, 1, n)]


def rain_cloud_plot_predictions_ALSPAC(df, flagPAD, group_col="pliks18TH",
                                       title=None, savepath=None):
    """Single-axis raincloud (ported from source line 242)."""
    df_plot = df.copy()
    group_order = sorted(df_plot[group_col].dropna().unique())
    df_plot = df_plot[df_plot[group_col].isin(group_order)]

    grey_light, grey_dark = "#B4BBBB", "#858E8D"
    red_light,  red_dark  = "#BF6673", "#C42840"

    if len(group_order) == 2:
        violin_palette = [grey_dark, red_dark]
        strip_palette  = [grey_light, red_light]
        box_palette    = _interp(grey_light, red_light, 2)
    else:
        violin_palette = sns.color_palette("Set2", len(group_order))
        strip_palette  = sns.color_palette("Set2", len(group_order))
        box_palette    = [_rgb_to_hex(c) for c in sns.color_palette("Set2", len(group_order))]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df_plot, x=group_col, y=flagPAD, order=group_order,
                   palette=violin_palette, linewidth=3.5, fill=False,
                   ax=ax, inner=None)
    for i, g in enumerate(group_order):
        sub = df_plot[df_plot[group_col] == g]
        sns.boxplot(data=sub, x=group_col, y=flagPAD, order=group_order,
                    width=0.5,
                    boxprops=dict(facecolor="none", edgecolor=box_palette[i], linewidth=2),
                    whiskerprops=dict(linewidth=2, color=box_palette[i]),
                    capprops=dict(linewidth=2, color=box_palette[i]),
                    medianprops=dict(linewidth=2, color=box_palette[i]),
                    showfliers=False, ax=ax)
    sns.stripplot(data=df_plot, x=group_col, y=flagPAD, order=group_order,
                  jitter=True, size=7, palette=strip_palette,
                  alpha=0.4, linewidth=0, ax=ax)

    for i, g in enumerate(group_order):
        m = df_plot.loc[df_plot[group_col] == g, flagPAD].mean()
        ax.scatter(i, m, color="black", s=70, zorder=3)
        ax.text(i, m + 0.1, f"Mean: {m:.2f}", ha="center", size="small",
                bbox=dict(facecolor="white", edgecolor="black",
                          boxstyle="round,pad=0.3"))

    ax.set_xlabel(group_col, fontweight="bold")
    ax.set_ylabel(flagPAD, fontweight="bold")
    ax.set_title(title or f"Raincloud: {flagPAD} by {group_col}",
                 fontweight="bold")
    if savepath:
        fig.savefig(savepath)
    plt.close(fig)


def rain_cloud_plot_comparison(df, color, flagPAD, group_col="pliks18TH",
                               savepath=None, title="Raincloud Comparison"):
    """Two-panel (control vs lumped PE | full 4-level) raincloud
    - ported from source line 365."""
    df_plot = df.copy()
    df_plot[group_col] = df_plot[group_col].astype(int)
    df_plot["__combined"] = df_plot[group_col].apply(lambda x: x if x == 0 else 123)
    order_combined = [0, 123]
    order_full     = sorted(df_plot[group_col].unique())

    if color == 1:
        violin_palette_left  = ["#B4BBBB", "#B80A24"]
        strip_palette_left   = ["#B4BBBB", "#B80A24"]
        violin_palette_right = ["#B4BBBB", "#C9919A", "#C96764", "#B80A24"]
    elif color == 2:
        violin_palette_left  = ["#6D89AD", "#B86609"]
        strip_palette_left   = ["#6D89AD", "#B86609"]
        violin_palette_right = ["#6D89AD", "#C9AB91", "#C99965", "#B86609"]
    else:
        violin_palette_left  = ["#866DAD", "#4CB809"]
        strip_palette_left   = ["#866DAD", "#4CB809"]
        violin_palette_right = ["#866DAD", "#91C9AC", "#6FC965", "#4CB809"]
    strip_palette_right = violin_palette_right

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(title, fontweight="bold", fontsize=16)

    # LEFT: 2 groups
    sns.violinplot(data=df_plot, x="__combined", y=flagPAD, order=order_combined,
                   palette=violin_palette_left, linewidth=3.5, fill=False,
                   ax=ax_l, inner=None)
    for i, g in enumerate(order_combined):
        sub = df_plot[df_plot["__combined"] == g]
        sns.boxplot(data=sub, x="__combined", y=flagPAD, order=order_combined,
                    width=0.5,
                    boxprops=dict(facecolor="none",
                                  edgecolor=violin_palette_left[i], linewidth=2),
                    whiskerprops=dict(linewidth=2, color=violin_palette_left[i]),
                    capprops=dict(linewidth=2, color=violin_palette_left[i]),
                    medianprops=dict(linewidth=2, color=violin_palette_left[i]),
                    showfliers=False, ax=ax_l)
    sns.stripplot(data=df_plot, x="__combined", y=flagPAD, order=order_combined,
                  jitter=True, size=7, palette=strip_palette_left,
                  alpha=0.4, linewidth=0, ax=ax_l)
    for i, g in enumerate(order_combined):
        m = df_plot.loc[df_plot["__combined"] == g, flagPAD].mean()
        ax_l.scatter(i, m, color="black", s=70, zorder=3)
        ax_l.text(i, m + 0.1, f"Mean: {m:.2f}", ha="center", size="small",
                  bbox=dict(facecolor="white", edgecolor="black",
                            boxstyle="round,pad=0.3"))
    labels = [f"{g}\nN={len(df_plot[df_plot['__combined'] == g])}"
              for g in order_combined]
    ax_l.set_xticks(range(len(order_combined)))
    ax_l.set_xticklabels(labels)
    ax_l.set_xlabel(f"{group_col} combined", fontweight="bold")
    ax_l.set_ylabel(flagPAD, fontweight="bold")
    ax_l.set_title("0 vs (1,2,3) combined", fontweight="bold")

    # RIGHT: 4 groups
    sns.violinplot(data=df_plot, x=group_col, y=flagPAD, order=order_full,
                   palette=violin_palette_right, linewidth=3.5, fill=False,
                   ax=ax_r, inner=None)
    for i, g in enumerate(order_full):
        sub = df_plot[df_plot[group_col] == g]
        sns.boxplot(data=sub, x=group_col, y=flagPAD, order=order_full,
                    width=0.5,
                    boxprops=dict(facecolor="none",
                                  edgecolor=violin_palette_right[i], linewidth=2),
                    whiskerprops=dict(linewidth=2, color=violin_palette_right[i]),
                    capprops=dict(linewidth=2, color=violin_palette_right[i]),
                    medianprops=dict(linewidth=2, color=violin_palette_right[i]),
                    showfliers=False, ax=ax_r)
    sns.stripplot(data=df_plot, x=group_col, y=flagPAD, order=order_full,
                  jitter=True, size=7, palette=strip_palette_right,
                  alpha=0.4, linewidth=0, ax=ax_r)
    for i, g in enumerate(order_full):
        m = df_plot.loc[df_plot[group_col] == g, flagPAD].mean()
        ax_r.scatter(i, m, color="black", s=70, zorder=3)
        ax_r.text(i, m + 0.1, f"Mean: {m:.2f}", ha="center", size="small",
                  bbox=dict(facecolor="white", edgecolor="black",
                            boxstyle="round,pad=0.3"))
    labels_full = [f"{g}\nN={len(df_plot[df_plot[group_col] == g])}"
                   for g in order_full]
    ax_r.set_xticks(range(len(order_full)))
    ax_r.set_xticklabels(labels_full)
    ax_r.set_xlabel(group_col, fontweight="bold")
    ax_r.set_ylabel(flagPAD, fontweight="bold")
    ax_r.set_title("Original groups (0,1,2,3)", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if savepath:
        fig.savefig(savepath)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pred_I_path  = f"{DATA_DIR}/preds_ALSPAC_I_with_covariates.csv"
    pred_II_path = f"{DATA_DIR}/preds_ALSPAC_II_with_covariates.csv"
    if os.path.exists(pred_I_path):
        pred_I = pd.read_csv(pred_I_path)
    else:
        pred_I = pd.read_csv(f"{DATA_DIR}/predictions_ALSPAC_20_MRI_I.csv")
    if os.path.exists(pred_II_path):
        pred_II = pd.read_csv(pred_II_path)
    else:
        pred_II = pd.read_csv(f"{DATA_DIR}/predictions_ALSPAC_30_MRI_II.csv")

    df_deltas_path = f"{DATA_DIR}/lmm_input_deltas.csv"
    df_deltas = pd.read_csv(df_deltas_path) if os.path.exists(df_deltas_path) else None

    results_path = f"{DATA_DIR}/lmm_results_main.csv"
    if os.path.exists(results_path):
        res = pd.read_csv(results_path)
        print("Loaded LMM results:", res.shape)
    else:
        res = None
        print("[warn] lmm_results_main.csv not found, q-values will not be overlaid")

    # Figure 5 - wave I and wave II raincloud across PE groups
    rain_cloud_plot_comparison(
        pred_I, color=1, flagPAD=PAD_COL,
        savepath=f"{FIG_DIR}/Figure5_raincloud_waveI.svg",
        title="Raincloud BrainPAD wave I (LPEs-1)",
    )
    rain_cloud_plot_comparison(
        pred_II, color=2, flagPAD=PAD_COL,
        savepath=f"{FIG_DIR}/Figure5_raincloud_waveII.svg",
        title="Raincloud BrainPAD wave II (LPEs-1)",
    )

    # Figure 6 - ΔBrainPAD raincloud across LPEs-1 and LPEs-2
    if df_deltas is not None:
        rain_cloud_plot_comparison(
            df_deltas, color=3, flagPAD=PAD_COL,
            savepath=f"{FIG_DIR}/Figure6_delta_raincloud_LPEs1.svg",
            title=u"ΔBrainPAD by LPEs-1 (pliks18TH)",
        )
        # LPEs-2: switch the grouping column to trajectory (4 levels).
        df_traj = df_deltas.dropna(subset=["trajectory"]).copy()
        df_traj.drop(columns=["pliks18TH"], inplace=True, errors="ignore")
        df_traj.rename(columns={"trajectory": "pliks18TH"}, inplace=True)
        df_traj["pliks18TH"] = df_traj["pliks18TH"].astype(int)
        rain_cloud_plot_comparison(
            df_traj, color=3, flagPAD=PAD_COL,
            savepath=f"{FIG_DIR}/Figure6_delta_raincloud_LPEs2.svg",
            title=u"ΔBrainPAD by LPEs-2 (trajectory)",
        )
    else:
        print("[warn] lmm_input_deltas.csv not found; Figure 6 skipped.")

    print("Done. Figure 4 (predicted-age trajectories) is produced by "
          "06_figure3_predictions.py - run that script directly.")


if __name__ == "__main__":
    main()
