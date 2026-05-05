"""
06_figure3_predictions.py
=========================
Figure 3 - Predicted vs Actual age plots.

Generates panels A (TestSet), B (AgeRisk), and C (combined ALSPAC-I + ALSPAC-II
coloured by PE category) from pre-computed prediction CSVs.

Run from the `src/` directory; all paths are relative to repo root.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import linregress, pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample

plt.rcParams.update({
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------
def bootstrap_mae(y_true, y_pred, n_boot=5000):
    maes = []
    for _ in range(n_boot):
        idx = resample(range(len(y_true)))
        maes.append(np.mean(np.abs(y_true[idx] - y_pred[idx])))
    return np.percentile(maes, [2.5, 97.5])


def bootstrap_r2(y_true, y_pred, n_boot=5000, ci=95, random_state=None):
    """Bootstrap CI for R^2."""
    rng = np.random.default_rng(random_state)
    s = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        s.append(r2_score(y_true[idx], y_pred[idx]))
    alpha = (100 - ci) / 2
    return np.percentile(s, [alpha, 100 - alpha])


def bootstrap_pearsonr(y_true, y_pred, n_boot=5000, ci=95, random_state=None):
    """Bootstrap CI for Pearson r."""
    rng = np.random.default_rng(random_state)
    s = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        s.append(np.corrcoef(y_true[idx], y_pred[idx])[0, 1])
    alpha = (100 - ci) / 2
    return np.percentile(s, [alpha, 100 - alpha])


# ---------------------------------------------------------------------------
# Sliding-window MAE plot
# ---------------------------------------------------------------------------
def plot_mae_sliding_windows(df, real_col='Edad', pred_col='pred_Edad',
                             window_size=3, overlap=5):
    """MAE of age predictions in overlapping age-windows."""
    if real_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{real_col}' and '{pred_col}' columns.")

    df_clean = df.dropna(subset=[real_col, pred_col]).copy()
    df_clean[real_col] = pd.to_numeric(df_clean[real_col], errors='coerce')
    df_clean[pred_col] = pd.to_numeric(df_clean[pred_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[real_col, pred_col])

    min_age = int(df_clean[real_col].min())
    max_age = int(df_clean[real_col].max())

    window_starts, mae_values = [], []
    for start_age in range(min_age, max_age - window_size + 2, overlap):
        end_age = start_age + window_size
        w = df_clean[(df_clean[real_col] >= start_age) & (df_clean[real_col] < end_age)]
        if len(w) == 0:
            continue
        window_starts.append(start_age)
        mae_values.append(mean_absolute_error(w[real_col], w[pred_col]))

    plt.figure(figsize=(8, 5))
    plt.plot(window_starts, mae_values, marker='o', linestyle='-', color='blue')
    plt.title('MAE for Age Prediction in Sliding Windows')
    plt.xlabel(f'Start Age of {window_size}-Year Window')
    plt.ylabel('MAE')
    plt.grid(True)
    plt.tight_layout()


# ---------------------------------------------------------------------------
# BrainPAD vs Age (raw + corrected)
# ---------------------------------------------------------------------------
def plot_brainpad_vs_age(
    results,
    age_col: str = "Edad",
    pad_cols: tuple = ("BrainPAD", "BrainPAD_c"),
    time_col: str = "Time",
    palette="deep",
):
    """Two square scatter plots: raw vs corrected BrainPAD, coloured by `time_col`."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    titles = ["BrainPAD vs. Age (raw)", "BrainPAD vs. Age (corrected)"]
    sns.set_palette(palette)

    for ax, y_col, title in zip(axes, pad_cols, titles):
        r, p_r = pearsonr(results[y_col], results[age_col])
        slope, intercept, r_val, p_slope, _ = linregress(results[age_col], results[y_col])
        print(f"[{y_col}]  r={r:.3f} (p={p_r:.3g})   "
              f"slope={slope:.3f}, R^2={r_val**2:.3f} (p={p_slope:.3g})")

        kwargs = dict(data=results, x=age_col, y=y_col, ax=ax,
                      s=100, alpha=1, linewidth=0.2, edgecolor="k")
        if time_col is not None and time_col in results.columns:
            kwargs["hue"] = time_col
        sns.scatterplot(**kwargs)

        x_fit = np.linspace(10, 40, 100)
        ax.plot(x_fit, intercept + slope * x_fit, color="green", lw=3,
                label="Linear fit", zorder=3)
        ax.axhline(0, color="k", ls="--", lw=1)

        ax.set_aspect("equal", adjustable="box")
        x_min, x_max = results[age_col].min(), results[age_col].max()
        y_min, y_max = results[y_col].min(), results[y_col].max()
        max_range = max(x_max - x_min, y_max - y_min)
        ax.set_xlim(((x_max + x_min) / 2) - (max_range / 2) - 5,
                    ((x_max + x_min) / 2) + (max_range / 2) + 5)
        ax.set_ylim(((y_max + y_min) / 2) - (max_range / 2) - 5,
                    ((y_max + y_min) / 2) + (max_range / 2) + 5)

        ax.set_xlabel("Chronological age (years)", fontsize=12)
        ax.set_ylabel("BrainPAD (years)", fontsize=12)
        ax.set_title(title, fontsize=14)

        r_patch = Patch(facecolor="none", edgecolor="none",
                        label=f"Pearson r = {r:.2f}")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(r_patch)
        labels.append(r_patch.get_label())
        ax.legend(handles, labels, loc="upper left", frameon=True,
                  fontsize=10, title=None)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Predicted vs Actual age - single scatter
# ---------------------------------------------------------------------------
def plot_age_vs_predicted(real_age, pred_age, label, fname_out=None):
    """
    Scatter of predicted vs real age + ideal-fit line. Reports MAE / r / R^2
    plus bootstrap CIs on stdout, and saves an SVG if `fname_out` is given.
    """
    real_age = np.asarray(real_age)
    pred_age = np.asarray(pred_age)
    print(f"[{label}] n={len(real_age)}")

    mae = mean_absolute_error(real_age, pred_age)
    r2 = r2_score(real_age, pred_age)
    r = stats.pearsonr(real_age, pred_age)[0]

    ic_low, ic_high = bootstrap_mae(real_age, pred_age, n_boot=2000)
    ic_r2_low, ic_r2_high = bootstrap_r2(real_age, pred_age, n_boot=5000)
    ic_r_low, ic_r_high = bootstrap_pearsonr(real_age, pred_age, n_boot=5000)

    fig = plt.figure(figsize=(8, 8))
    sns.scatterplot(x=real_age, y=pred_age, s=140, edgecolor="k",
                    color="#1f77b4", linewidth=.5, alpha=0.8)
    plt.plot([0, 60], [0, 60], 'k--', lw=2, label='Ideal Fit')
    plt.xlabel('Real Age', fontsize=20)
    plt.ylabel('Predicted Age', fontsize=20)
    plt.title(f'Predicted vs. Real Age - {label}')
    plt.xlim(0, 60); plt.ylim(0, 60)
    plt.xticks(fontsize=18); plt.yticks(fontsize=18)
    plt.gca().set_aspect('equal', adjustable='box')

    print(f'CI MAE: {(round(ic_low, 2), round(ic_high, 2))}')
    print(f'CI R^2: {(round(ic_r2_low, 2), round(ic_r2_high, 2))}')
    print(f'CI r:   {(round(ic_r_low, 2), round(ic_r_high, 2))}')

    textstr = '\n'.join((
        f'MAE: {mae:.2f}',
        f'Pearson r: {r:.2f}',
        f'R^2: {r2:.2f}'))
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if fname_out is not None:
        plt.savefig(fname_out)
    return fig


# ---------------------------------------------------------------------------
# Predicted vs Actual age - multi-group scatter (PE / DataBase)
# ---------------------------------------------------------------------------
def plot_age_vs_pred(data, real_col="Edad", pred_col="pred_Edad_c",
                     pliks_col="pliks18TH", acquisition_col="DataBase",
                     fname_out="fig_age_pred.svg"):
    """
    Scatter coloured by PE level (`pliks18TH`) and styled by acquisition wave
    (`DataBase`). Saves SVG to `fname_out`.
    """
    plot_brainpad_vs_age(data)

    pliks_map = {
        0: "Controls",
        1: "Suspected",
        2: "Definite",
        3: "Clinical Disorder",
    }
    pal = sns.color_palette("tab10")
    hue_order = ['Controls', 'Definite', 'Suspected', 'Clinical Disorder']
    hue2rgb = {h: pal[i % len(pal)] for i, h in enumerate(hue_order)}

    data = data.copy()
    data["pliks18TH_label"] = data["pliks18TH"].map(pliks_map)
    pliks_col = "pliks18TH_label"

    db_map = {
        "subjects_ALSPAC": "ALSPAC 20 MRI-I",
        "subjects_ALSPAC_II_adq": "ALSPAC 30 MRI-II",
    }
    data["DataBase_label"] = data["DataBase"].map(db_map)
    acquisition_col = "DataBase_label"

    mae = mean_absolute_error(data[real_col], data[pred_col])
    r = stats.pearsonr(data[real_col], data[pred_col])[0]
    r2 = r2_score(data[real_col], data[pred_col])

    fig = plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=data, x=real_col, y=pred_col,
        hue=pliks_col, style=acquisition_col,
        palette=hue2rgb, s=140, edgecolor="k",
        linewidth=.5, alpha=0.7,
    )
    plt.plot([5, 60], [5, 60], "k--", lw=1, label="Ideal fit")
    plt.xlim(5, 60); plt.ylim(5, 60)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xticks(fontsize=18); plt.yticks(fontsize=18)
    plt.xlabel("Chronological age (years)", fontsize=20)
    plt.ylabel("Predicted age (years)", fontsize=20)
    plt.title("Predicted vs chronological age")

    txt = (f"MAE = {mae:.2f}\n"
           f"Pearson r = {r:.2f}\n"
           f"R^2 = {r2:.2f}")
    plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes, va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.legend(title="PE level / Acquisition", loc="lower right")
    plt.tight_layout()
    plt.savefig(fname_out)
    return fig


# ---------------------------------------------------------------------------
# Predicted vs Actual age - longitudinal (wave I -> wave II)
# ---------------------------------------------------------------------------
def plot_age_vs_predicted_longitudinal(
    data: pd.DataFrame,
    id_col: str = "ID",
    real_col: str = "Edad",
    pred_col: str = "pred_Edad_c",
    pliks_col: str = "pliks18TH",
    acquisition_col: str = "DataBase",
    wave1: str = "T1",
    wave2: str = "T2",
    fname_out: str = "fig_age_pred_longitudinal.svg",
):
    """
    Scatter of predicted vs real age at two waves with line segments connecting
    the same subjects across waves. Circle = wave1, X = wave2; colour = PE.
    """
    pliks_map_1 = {
        0: "Controls",
        1: "Suspected",
        2: "Definite",
        3: "Clinical Disorder",
    }

    data = data.copy()
    data["pliks18TH_label"] = data["pliks18TH"].map(pliks_map_1)
    pliks_col = "pliks18TH_label"

    data[id_col] = data[id_col].astype(str).str.replace('_brain', '', regex=False)\
                                            .str.replace('sub-', '', regex=False)
    counts = data[id_col].value_counts()
    repeated_ids = counts[counts > 1].index
    data = data[data[id_col].isin(repeated_ids)]

    pal = sns.color_palette("tab10")
    hue_order = ['Controls', 'Definite', 'Suspected', 'Clinical Disorder']
    hue2rgb = {h: pal[i % len(pal)] for i, h in enumerate(hue_order)}

    mae = mean_absolute_error(data[real_col], data[pred_col])
    r = stats.pearsonr(data[real_col], data[pred_col])[0]
    r2 = r2_score(data[real_col], data[pred_col])

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(
        data=data, x=real_col, y=pred_col,
        hue=pliks_col, style=acquisition_col,
        palette=hue2rgb, markers={wave1: "o", wave2: "X"},
        s=140, edgecolor="k", linewidth=.5, alpha=.7, ax=ax,
    )

    d1 = data.loc[data[acquisition_col] == wave1].copy()
    d2 = data.loc[data[acquisition_col] == wave2].copy()
    both = pd.merge(
        d1[[id_col, real_col, pred_col, pliks_col]],
        d2[[id_col, real_col, pred_col]],
        on=id_col, suffixes=("_1", "_2"),
    )
    for _, row in both.iterrows():
        c = hue2rgb.get(row[pliks_col], "grey")
        ax.plot(
            [row[f"{real_col}_1"], row[f"{real_col}_2"]],
            [row[f"{pred_col}_1"], row[f"{pred_col}_2"]],
            lw=0.8, color=c, alpha=.6,
        )

    ax.plot([15, 40], [15, 40], "k--", lw=1, label="Ideal fit")
    ax.set_xlim(18, 32); ax.set_ylim(5, 55)
    plt.xticks(fontsize=18); plt.yticks(fontsize=18)
    ax.set_box_aspect(1)
    ax.set_xlabel("Real Age", fontsize=20)
    ax.set_ylabel("Predicted age", fontsize=20)
    ax.set_title("Predicted vs chronological age - longitudinal")

    txt = (f"MAE = {mae:.2f}\n"
           f"Pearson r = {r:.2f}\n"
           f"R^2 = {r2:.2f}")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend(title="PE / Wave", loc="lower right")
    fig.tight_layout()
    fig.savefig(fname_out)
    return fig


# ---------------------------------------------------------------------------
# Driver - Figure 3 panels A, B, C
# ---------------------------------------------------------------------------
def main():
    data_dir = "../data"
    fig_dir = "../figures"

    # Load pre-computed predictions
    result_TestSet = pd.read_csv(f"{data_dir}/predictions_TestSet.csv")
    result_AgeRisk = pd.read_csv(f"{data_dir}/predictions_AgeRisk.csv")
    result_alspac_I = pd.read_csv(f"{data_dir}/predictions_ALSPAC_20_MRI_I.csv")
    result_alspac_II = pd.read_csv(f"{data_dir}/predictions_ALSPAC_30_MRI_II.csv")

    # Panel A: TestSet
    plot_age_vs_predicted(
        result_TestSet["Edad"].values,
        result_TestSet["pred_Edad"].values,
        label="TestSet",
        fname_out=f"{fig_dir}/Figure3A_TestSet.svg",
    )

    # Panel B: AgeRisk
    plot_age_vs_predicted(
        result_AgeRisk["Edad"].values,
        result_AgeRisk["pred_Edad"].values,
        label="AgeRisk",
        fname_out=f"{fig_dir}/Figure3B_AgeRisk.svg",
    )

    # Panel C: combined ALSPAC-I + ALSPAC-II
    result_alspac_I = result_alspac_I.copy()
    result_alspac_II = result_alspac_II.copy()
    result_alspac_I["Time"] = "ALSPAC-20 MRI-I"
    result_alspac_II["Time"] = "ALSPAC-30 MRI-II"

    for df in (result_alspac_I, result_alspac_II):
        if "ID" in df.columns:
            df["ID"] = (df["ID"].astype(str)
                                 .str.replace("_brain", "", regex=False)
                                 .str.replace("sub-", "", regex=False))

    results_combined = pd.concat([result_alspac_I, result_alspac_II], axis=0)

    # PE-trajectory category (used for colouring)
    if "pliks30TH" in results_combined.columns:
        conditions = [
            (results_combined["pliks18TH"] == 0) & (results_combined["pliks30TH"] == 0),
            (results_combined["pliks18TH"].isin([1, 2, 3])) & (results_combined["pliks30TH"] == 0),
            (results_combined["pliks18TH"].isin([1, 2, 3])) & (results_combined["pliks30TH"].isin([1, 2, 3])),
            (results_combined["pliks18TH"] == 0) & (results_combined["pliks30TH"].isin([1, 2, 3])),
        ]
        choices = [0, 1, 2, 3]
        results_combined["trajectory"] = np.select(conditions, choices, default=np.nan)

    plot_age_vs_pred(
        results_combined,
        real_col="Edad",
        pred_col="pred_Edad_c",
        pliks_col="pliks18TH",
        acquisition_col="DataBase",
        fname_out=f"{fig_dir}/Figure3C_ALSPAC.svg",
    )

    # Longitudinal companion plot
    plot_age_vs_predicted_longitudinal(
        data=results_combined,
        id_col="ID",
        acquisition_col="Time",
        wave1="ALSPAC-20 MRI-I",
        wave2="ALSPAC-30 MRI-II",
        fname_out=f"{fig_dir}/Figure3C_ALSPAC_longitudinal.svg",
    )

    plt.show()


if __name__ == "__main__":
    main()
