"""
04_descriptive_stats.py
-----------------------
Builds Table 1 (cross-sectional demographics for ALSPAC-20 / MRI-I and
ALSPAC-30 / MRI-II by PE category) and Table S1 (longitudinal subset by
LPEs-1 = pliks18TH categories and LPEs-2 = trajectory class).

Stats reported in the paper:
    - Mann-Whitney U on age (Controls vs PE+)
    - chi-square on sex
For multi-category breakdowns the script also reports Kruskal-Wallis
(age) and chi-square (sex) across all groups.

Run from `src/`:
    python 04_descriptive_stats.py
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats

try:
    from tabulate import tabulate
    HAVE_TAB = True
except ImportError:
    HAVE_TAB = False


# ---------------------------------------------------------------- paths ----
PRED_I_PATH      = '../data/predictions_ALSPAC_20_MRI_I.csv'
PRED_II_PATH     = '../data/predictions_ALSPAC_30_MRI_II.csv'
DEPR_I_PATH      = '../data/ALSPAC_I_depression.csv'   # user must supply
DEPR_II_PATH     = '../data/ALSPAC_II_depression.csv'  # user must supply
OUT_DIR          = '../figures'
os.makedirs(OUT_DIR, exist_ok=True)

IDS_TO_REMOVE = ['SUBJ_00432', 'SUBJ_00557', 'SUBJ_00579',
                 'SUBJ_00351', 'SUBJ_00501']

# Canonical labels matching the paper
PLIKS_LABELS = {0: 'Controls', 1: 'Suspected', 2: 'Definite', 3: 'Clinical Disorder'}
TRAJ_LABELS  = {0: 'Controls', 1: 'Remitted', 2: 'Persistent', 3: 'Incident'}


# ---------------------------------------------------------------- helpers --
def _clean_id(s):
    return s.str.replace('_brain', '', regex=False).str.replace('sub-', '', regex=False)


def fmt_mean_sd(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return 'NA'
    return f"{x.mean():.2f} +/- {x.std():.2f}"


def sex_counts(s, female_codes=(0, 'F', 'Female', 'female')):
    """Return 'F/M (xx.x% F)' string. Assumes column 'sexo' codes M=1, F=0
    (matches the convention 'sexo(M=1;F=0)' used throughout the pipeline).
    """
    s = pd.Series(s).dropna()
    n_total = len(s)
    if n_total == 0:
        return 'NA'
    n_female = int(s.isin(female_codes).sum())
    n_male = n_total - n_female
    pct_f = 100.0 * n_female / n_total if n_total else 0.0
    return f"{n_female}/{n_male} ({pct_f:.1f}% F)"


def mwu_p(values_a, values_b):
    a = pd.Series(values_a).dropna()
    b = pd.Series(values_b).dropna()
    if len(a) < 1 or len(b) < 1:
        return np.nan, np.nan
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    return u, p


def kruskal_p(*groups):
    groups = [pd.Series(g).dropna() for g in groups if len(pd.Series(g).dropna()) > 0]
    if len(groups) < 2:
        return np.nan, np.nan
    h, p = stats.kruskal(*groups)
    return h, p


def chi2_p(df, group_col, target_col='sexo'):
    tab = pd.crosstab(df[group_col], df[target_col])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan, np.nan
    chi2, p, _, _ = stats.chi2_contingency(tab)
    return chi2, p


def scanner_counts(s):
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return 'NA'
    parts = [f"{k}: {v}" for k, v in s.value_counts().sort_index().items()]
    return '; '.join(parts)


# ---------------------------------------------------------------- load -----
def load_predictions():
    preds_I  = pd.read_csv(PRED_I_PATH)
    preds_II = pd.read_csv(PRED_II_PATH)
    preds_I['ID']  = _clean_id(preds_I['ID'])
    preds_II['ID'] = _clean_id(preds_II['ID'])

    # Optional merge with depression CSVs (kept for compatibility w/ original).
    try:
        d_I  = pd.read_csv(DEPR_I_PATH)
        d_II = pd.read_csv(DEPR_II_PATH)
        d_I.rename(columns={'id': 'ID'}, inplace=True)
        d_II['ID'] = _clean_id(d_II['ID'].astype(str))
        cols_I  = [c for c in ['ID', 'FJCI1001', 'FJCI1002', 'FJCI501', 'FKSH1070', 'FJCI350'] if c in d_I.columns]
        cols_II = [c for c in ['ID', 'pliks30TH', 'FJCI1001', 'FJCI1002', 'FJCI501', 'FKSH1070', 'FJCI350'] if c in d_II.columns]
        preds_I  = preds_I.merge(d_I[cols_I],   on='ID', how='left')
        preds_II = preds_II.merge(d_II[cols_II], on='ID', how='left')
    except FileNotFoundError:
        print('[warn] depression CSVs not found - skipping merge.')

    preds_I['Time']  = 0
    preds_II['Time'] = 1
    if 'pliks18TH' in preds_I.columns:
        preds_I['pliks18TH'] = preds_I['pliks18TH'].fillna(0)
    if 'pliks18TH' in preds_II.columns:
        preds_II['pliks18TH'] = preds_II['pliks18TH'].fillna(0)
    preds_I = preds_I[~preds_I['ID'].isin(IDS_TO_REMOVE)]

    preds_I['pliks_bin']  = (preds_I['pliks18TH']  > 0.5).astype(int)
    preds_II['pliks_bin'] = (preds_II['pliks18TH'] > 0.5).astype(int)
    return preds_I, preds_II


def build_longitudinal(preds_I, preds_II):
    """Subjects scanned at BOTH time points."""
    both = pd.concat([preds_I, preds_II], axis=0, ignore_index=True)
    counts = both['ID'].value_counts()
    keep = counts[counts == 2].index
    both = both[both['ID'].isin(keep)].reset_index(drop=True)

    # Optional restriction to subjects_ALSPAC database
    if 'DataBase' in both.columns:
        both = both[both['DataBase'] == 'subjects_ALSPAC'].reset_index(drop=True)

    # ---- LPEs-1: baseline pliks18TH categories (0/1/2/3) ----
    base = both[both['Time'] == 0].copy()
    follow = both[both['Time'] == 1].copy()

    # ---- LPEs-2: trajectory using pliks18TH (T0) and pliks30TH (T1) ----
    if 'pliks30TH' in both.columns:
        b = base[['ID', 'pliks18TH', 'Edad', 'sexo'] + (['Escaner'] if 'Escaner' in base.columns else [])].copy()
        f = follow[['ID', 'pliks30TH']].copy()
        traj_df = b.merge(f, on='ID', how='inner')
        traj_df['pliks30TH'] = traj_df['pliks30TH'].astype('Float64').fillna(-1).astype('Int64')
        traj_df['pliks18TH'] = traj_df['pliks18TH'].astype('Float64').fillna(0).astype('Int64')

        conditions = [
            (traj_df['pliks18TH'] == 0) & (traj_df['pliks30TH'] == 0),
            (traj_df['pliks18TH'].isin([1, 2, 3])) & (traj_df['pliks30TH'] == 0),
            (traj_df['pliks18TH'].isin([1, 2, 3])) & (traj_df['pliks30TH'].isin([1, 2, 3])),
            (traj_df['pliks18TH'] == 0) & (traj_df['pliks30TH'].isin([1, 2, 3])),
        ]
        choices = [0, 1, 2, 3]   # Controls, Remitted, Persistent, Incident
        traj_df['trajectory'] = np.select(conditions, choices, default=np.nan)
    else:
        traj_df = pd.DataFrame()

    return base, follow, traj_df


# ---------------------------------------------------------------- tables ---
def cross_sectional_table(df, cohort_label):
    """Build one block of Table 1 for a single cohort."""
    df = df.copy()
    df['pliks18TH'] = df['pliks18TH'].astype('Int64', errors='ignore').fillna(0).astype(int)
    df['pliks_bin'] = (df['pliks18TH'] > 0).astype(int)

    rows = []
    for code in [0, 1, 2, 3]:
        sub = df[df['pliks18TH'] == code]
        rows.append({
            'Cohort':   cohort_label,
            'Group':    PLIKS_LABELS[code],
            'N':        len(sub),
            'Age (mean +/- SD)': fmt_mean_sd(sub['Edad']) if 'Edad' in sub.columns else 'NA',
            'Sex (F/M)':         sex_counts(sub['sexo']) if 'sexo' in sub.columns else 'NA',
            'Scanner':           scanner_counts(sub['Escaner']) if 'Escaner' in sub.columns else 'NA',
        })
    pe_pos = df[df['pliks_bin'] == 1]
    rows.append({
        'Cohort':   cohort_label,
        'Group':    'PEs (any)',
        'N':        len(pe_pos),
        'Age (mean +/- SD)': fmt_mean_sd(pe_pos['Edad']) if 'Edad' in pe_pos.columns else 'NA',
        'Sex (F/M)':         sex_counts(pe_pos['sexo']) if 'sexo' in pe_pos.columns else 'NA',
        'Scanner':           scanner_counts(pe_pos['Escaner']) if 'Escaner' in pe_pos.columns else 'NA',
    })

    table = pd.DataFrame(rows)

    ctrl = df[df['pliks_bin'] == 0]
    pe   = df[df['pliks_bin'] == 1]

    u_age, p_age_mwu = mwu_p(ctrl['Edad'], pe['Edad']) if 'Edad' in df.columns else (np.nan, np.nan)
    chi2_sex, p_sex  = chi2_p(df, 'pliks_bin', 'sexo') if 'sexo' in df.columns else (np.nan, np.nan)

    groups_age = [df.loc[df['pliks18TH'] == c, 'Edad'] for c in [0, 1, 2, 3] if (df['pliks18TH'] == c).any()]
    h_age, p_age_kw = kruskal_p(*groups_age) if 'Edad' in df.columns else (np.nan, np.nan)
    chi2_sex_all, p_sex_all = chi2_p(df, 'pliks18TH', 'sexo') if 'sexo' in df.columns else (np.nan, np.nan)

    stats_lines = [
        f"[{cohort_label}] Mann-Whitney U age (Controls vs PE+):  U={u_age:.2f}  p={p_age_mwu:.4g}",
        f"[{cohort_label}] Chi^2 sex (Controls vs PE+):           chi2={chi2_sex:.3f}  p={p_sex:.4g}",
        f"[{cohort_label}] Kruskal-Wallis age (4 groups):         H={h_age:.3f}  p={p_age_kw:.4g}",
        f"[{cohort_label}] Chi^2 sex (4 groups):                  chi2={chi2_sex_all:.3f}  p={p_sex_all:.4g}",
    ]
    return table, stats_lines


def longitudinal_table(base, traj_df):
    """Table S1: LPEs-1 + LPEs-2 demographics."""
    rows = []
    stats_lines = []

    if not base.empty and 'pliks18TH' in base.columns:
        b = base.copy()
        b['pliks18TH'] = b['pliks18TH'].astype('Int64', errors='ignore').fillna(0).astype(int)
        b['pliks_bin'] = (b['pliks18TH'] > 0).astype(int)
        for code in [0, 1, 2, 3]:
            sub = b[b['pliks18TH'] == code]
            rows.append({
                'Definition': 'LPEs-1 (PLIKS-18)',
                'Group':      PLIKS_LABELS[code],
                'N':          len(sub),
                'Age (mean +/- SD)': fmt_mean_sd(sub['Edad']) if 'Edad' in sub.columns else 'NA',
                'Sex (F/M)':         sex_counts(sub['sexo']) if 'sexo' in sub.columns else 'NA',
                'Scanner':           scanner_counts(sub['Escaner']) if 'Escaner' in sub.columns else 'NA',
            })
        pe_pos = b[b['pliks_bin'] == 1]
        rows.append({
            'Definition': 'LPEs-1 (PLIKS-18)',
            'Group':      'PEs (any)',
            'N':          len(pe_pos),
            'Age (mean +/- SD)': fmt_mean_sd(pe_pos['Edad']) if 'Edad' in pe_pos.columns else 'NA',
            'Sex (F/M)':         sex_counts(pe_pos['sexo']) if 'sexo' in pe_pos.columns else 'NA',
            'Scanner':           scanner_counts(pe_pos['Escaner']) if 'Escaner' in pe_pos.columns else 'NA',
        })

        u_age, p_age_mwu = mwu_p(b.loc[b['pliks_bin'] == 0, 'Edad'], b.loc[b['pliks_bin'] == 1, 'Edad'])
        chi2_sex, p_sex  = chi2_p(b, 'pliks_bin', 'sexo')
        stats_lines.append(f"[LPEs-1] Mann-Whitney U age (Controls vs PE+): U={u_age:.2f}  p={p_age_mwu:.4g}")
        stats_lines.append(f"[LPEs-1] Chi^2 sex (Controls vs PE+):           chi2={chi2_sex:.3f}  p={p_sex:.4g}")

    if not traj_df.empty and 'trajectory' in traj_df.columns:
        t = traj_df.copy()
        t['traj_bin'] = (t['trajectory'] != 0).astype(int)
        for code in [0, 1, 2, 3]:
            sub = t[t['trajectory'] == code]
            rows.append({
                'Definition': 'LPEs-2 (trajectory)',
                'Group':      TRAJ_LABELS[code],
                'N':          len(sub),
                'Age (mean +/- SD)': fmt_mean_sd(sub['Edad']) if 'Edad' in sub.columns else 'NA',
                'Sex (F/M)':         sex_counts(sub['sexo']) if 'sexo' in sub.columns else 'NA',
                'Scanner':           scanner_counts(sub['Escaner']) if 'Escaner' in sub.columns else 'NA',
            })
        pe_pos = t[t['traj_bin'] == 1]
        rows.append({
            'Definition': 'LPEs-2 (trajectory)',
            'Group':      'PEs (any trajectory)',
            'N':          len(pe_pos),
            'Age (mean +/- SD)': fmt_mean_sd(pe_pos['Edad']) if 'Edad' in pe_pos.columns else 'NA',
            'Sex (F/M)':         sex_counts(pe_pos['sexo']) if 'sexo' in pe_pos.columns else 'NA',
            'Scanner':           scanner_counts(pe_pos['Escaner']) if 'Escaner' in pe_pos.columns else 'NA',
        })

        u_age, p_age_mwu = mwu_p(t.loc[t['traj_bin'] == 0, 'Edad'], t.loc[t['traj_bin'] == 1, 'Edad'])
        chi2_sex, p_sex  = chi2_p(t, 'traj_bin', 'sexo')
        stats_lines.append(f"[LPEs-2] Mann-Whitney U age (Controls vs PE+): U={u_age:.2f}  p={p_age_mwu:.4g}")
        stats_lines.append(f"[LPEs-2] Chi^2 sex (Controls vs PE+):           chi2={chi2_sex:.3f}  p={p_sex:.4g}")

        groups_age = [t.loc[t['trajectory'] == c, 'Edad'] for c in [0, 1, 2, 3] if (t['trajectory'] == c).any()]
        h_age, p_age_kw = kruskal_p(*groups_age)
        chi2_sex_all, p_sex_all = chi2_p(t, 'trajectory', 'sexo')
        stats_lines.append(f"[LPEs-2] Kruskal-Wallis age (4 groups): H={h_age:.3f}  p={p_age_kw:.4g}")
        stats_lines.append(f"[LPEs-2] Chi^2 sex (4 groups):           chi2={chi2_sex_all:.3f}  p={p_sex_all:.4g}")

    return pd.DataFrame(rows), stats_lines


def pretty_print(title, df):
    print('\n' + '=' * len(title))
    print(title)
    print('=' * len(title))
    if HAVE_TAB:
        print(tabulate(df, headers='keys', tablefmt='github', showindex=False))
    else:
        print(df.to_string(index=False))


# ---------------------------------------------------------------- main -----
def main():
    preds_I, preds_II = load_predictions()

    # ---- Table 1 ----
    t1_I,  s1_I  = cross_sectional_table(preds_I,  'ALSPAC-20 / MRI-I')
    t1_II, s1_II = cross_sectional_table(preds_II, 'ALSPAC-30 / MRI-II')
    table1 = pd.concat([t1_I, t1_II], axis=0, ignore_index=True)

    pretty_print('Table 1 - Cross-sectional demographics by PE category', table1)
    print()
    for line in s1_I + s1_II:
        print(line)

    table1.to_csv(os.path.join(OUT_DIR, 'Table1_cross_sectional_demographics.csv'), index=False)

    # ---- Table S1 ----
    base, follow, traj_df = build_longitudinal(preds_I, preds_II)
    tableS1, sS1 = longitudinal_table(base, traj_df)

    pretty_print('Table S1 - Longitudinal demographics (LPEs-1 and LPEs-2)', tableS1)
    print()
    for line in sS1:
        print(line)

    tableS1.to_csv(os.path.join(OUT_DIR, 'TableS1_longitudinal_demographics.csv'), index=False)


if __name__ == '__main__':
    main()
