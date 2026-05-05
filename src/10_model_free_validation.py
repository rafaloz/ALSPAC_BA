#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-free validation: control-trained PCA age axis (canonical).

This script REPLACES the older PLS-based port (`Review_2_Experiment.py` from
ALSPAC_review). It is the canonical reviewer-2 analysis that produces the
published supplement Table S17 and Figure S6.

Reviewer-2 checks (with scanner diagnostics)
============================================

A) Unsupervised PCA "age axis" trained ONLY on STRICT longitudinal controls
   (PLIKS==0 at BOTH waves), with:
     - Per-PC Spearman(age, score): controls vs all; raw vs oriented (csv + barplot)
     - Selection of top-K PCs by |rho(age)| in controls
     - For each selected PC:
         * projection to full samples (I, II)
         * age-adjusted residuals using control-only quadratic fit (age + age^2)
         * PLIKS dose plots + Spearman(PLIKS, PC residual) (csv + console)
     - Fused (|rho|-weighted) axis kept (dose plot + stats)
     - Run on harmonized and (when available) unharmonized features
     - For ALL PCs (controls-only):
         * R2(Age) from OLS: PC ~ 1 + Age + Age^2
         * R2(+Scanner) from OLS: PC ~ 1 + Age + Age^2 + Scanner dummies
         * deltaR2, block F-test for adding Scanner, partial r(Age|Scanner)

B) Longitudinal Annualized % Change (APC) across ROIs:
     - Subject-level mean |APC|/yr
     - Spearman(PLIKS_t2, mean|APC|/yr) and OLS slope +/- SE
     - Delta-years comparability tables by PLIKS
     - Optional covariates (sex, eTIV) if present

Outputs go to: ../figures/rev2_checks/HARM and ../figures/rev2_checks/UNHARM
"""

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")

# ===================== paths =====================
OUTDIR = '../figures/rev2_checks'
os.makedirs(OUTDIR, exist_ok=True)

I_UNHARM  = '../data/ALSPAC_20_MRI_I_raw.csv'
II_UNHARM = '../data/ALSPAC_30_MRI_II_raw.csv'
I_HARMO  = '../data/ALSPAC_20_MRI_I_harmonized.csv'
II_HARMO = '../data/ALSPAC_30_MRI_II_harmonized.csv'

# ================ config =================
K_LIST   = [1, 2, 3, 4, 5]
MAX_PCS  = 50
APC_MODE = "Standard"

YLIMS_BY_DATASET = {
    "Unharmonized": (-4.0, 3.0),
    "ComBat-GAM":   (-5.0, 10.0),
}

META = {
    'ID','Edad','Patologia','sexo','sexo(M=1;F=0)','eTIV',
    'Escaner','Scanner','scanner','DataBase','Bo',
    'pliks18TH','pliks20TH','pliks30TH','Time'
}


def ensure_dir(path): os.makedirs(path, exist_ok=True)


def canon_id(s: str) -> str:
    s = str(s).strip()
    s = s.replace('sub-', '').replace('SUB-', '')
    s = s.replace('_brain', '')
    s = re.sub(r'\.nii(\.gz)?$', '', s, flags=re.I)
    return s.upper()


def fmt_p(p):
    if p is None or not np.isfinite(p): return "nan"
    if p < 1e-3: return f"{p:.2e}"
    return f"{p:.4f}"


def pick_pliks(df):
    cands = [x for x in df.columns if x.lower().startswith('pliks')]
    return cands if cands else None


def pick_scanner_col(df):
    for c in ['Escaner','Scanner','scanner']:
        if c in df.columns:
            return c
    return None


def rois_from_harmonized(dfH):
    return [c for c in dfH.columns if c not in META and pd.api.types.is_numeric_dtype(dfH[c])]


def z_impute_controls(X, mask_ctrl):
    med = np.nanmedian(X[mask_ctrl, :], axis=0)
    X_imp = np.where(np.isnan(X), med[None, :], X)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_imp[mask_ctrl, :])
    return X_imp, scaler, med


def benjamini_hochberg(p):
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty(n, int); ranks[order] = np.arange(1, n+1)
    q = p * n / ranks
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    q_adj = np.empty_like(q); q_adj[order] = q_sorted
    return np.minimum(q_adj, 1.0)


def age_adjust_residuals(score, age, mask_ctrl):
    resid = np.full_like(score, np.nan, dtype=float)
    m = np.isfinite(score) & np.isfinite(age)
    if not m.any():
        return resid, np.array([np.nan, np.nan, np.nan])
    mc = m & mask_ctrl
    if mc.sum() >= 20:
        Xc = np.column_stack([np.ones(mc.sum()), age[mc], age[mc]**2])
        b, *_ = np.linalg.lstsq(Xc, score[mc], rcond=None)
        Xa = np.column_stack([np.ones(m.sum()), age[m], age[m]**2])
        resid[m] = score[m] - Xa @ b
        return resid, b
    else:
        resid[m] = score[m] - np.nanmean(score[m])
        return resid, np.array([np.nan, np.nan, np.nan])


def backfill_pliks_columns(I_h, II_h, I_u, II_u):
    if I_u is not None and 'pliks18TH' not in I_h.columns and 'pliks18TH' in I_u.columns:
        I_h = I_h.merge(I_u[['ID','pliks18TH']].drop_duplicates('ID'), on='ID', how='left')

    if II_u is not None and 'pliks30TH' not in II_h.columns and 'pliks30TH' in II_u.columns:
        II_h = II_h.merge(II_u[['ID','pliks30TH']].drop_duplicates('ID'), on='ID', how='left')
        if I_u is not None:
            I_u  = I_u.merge(II_u[['ID','pliks30TH']].drop_duplicates('ID'), on='ID', how='left')
        I_h  = I_h.merge(II_u[['ID','pliks30TH']].drop_duplicates('ID'), on='ID', how='left')

    pairs = [(I_h,'pliks18TH'), (II_h,'pliks30TH')]
    if I_u is not None:  pairs.append((I_u,'pliks18TH'))
    if II_u is not None: pairs.append((II_u,'pliks30TH'))
    for df, col in pairs:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').clip(0,3)

    return I_h, II_h, I_u, II_u


def _ols_R2(y, X):
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if m.sum() < X.shape[1] + 2:
        return np.nan, np.zeros(X.shape[1])
    b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    yhat = X[m] @ b
    ssr  = np.sum((y[m] - yhat)**2)
    sst  = np.sum((y[m] - np.mean(y[m]))**2)
    R2   = 1.0 - ssr / sst if sst > 0 else np.nan
    return R2, b


def scanner_block_stats(score, age, scanner_labels, mask_ctrl):
    m = np.isfinite(score) & np.isfinite(age) & mask_ctrl
    if m.sum() < 20:
        return dict(R2_age=np.nan, R2_full=np.nan, deltaR2=np.nan,
                    F=np.nan, df1=np.nan, df2=np.nan, p_block=np.nan)

    y  = score[m]
    a  = age[m]
    X_age = np.column_stack([np.ones(m.sum()), a, a**2])

    sc = scanner_labels[m].astype(str)
    levels = np.unique(sc)
    if levels.size <= 1:
        R2_age, _ = _ols_R2(y, X_age)
        return dict(R2_age=R2_age, R2_full=R2_age, deltaR2=0.0,
                    F=np.nan, df1=0, df2=(m.sum()-X_age.shape[1]), p_block=np.nan)

    dummies = np.column_stack([(sc == lv).astype(float) for lv in levels[1:]])
    X_full  = np.column_stack([X_age, dummies])

    R2_age,  _ = _ols_R2(y, X_age)
    R2_full, _ = _ols_R2(y, X_full)

    p_age  = X_age.shape[1]
    p_full = X_full.shape[1]
    N      = m.sum()
    df1    = p_full - p_age
    df2    = N - p_full

    if not np.isfinite(R2_age) or not np.isfinite(R2_full) or df1 <= 0 or df2 <= 0:
        return dict(R2_age=R2_age, R2_full=R2_full, deltaR2=(R2_full-R2_age),
                    F=np.nan, df1=df1, df2=df2, p_block=np.nan)

    num = (R2_full - R2_age) / df1
    den = (1 - R2_full) / df2
    F   = num / den if den > 0 else np.nan
    p   = 1 - stats.f.cdf(F, df1, df2) if np.isfinite(F) else np.nan

    return dict(R2_age=R2_age, R2_full=R2_full, deltaR2=(R2_full-R2_age),
                F=F, df1=df1, df2=df2, p_block=p)


def partial_corr_age_given_scanner(score, age, scanner_labels, mask_ctrl):
    m = np.isfinite(score) & np.isfinite(age) & mask_ctrl
    if m.sum() < 20:
        return np.nan, np.nan, np.nan

    sc = scanner_labels[m].astype(str)
    levels = np.unique(sc)
    if levels.size <= 1:
        Xs = np.ones((m.sum(), 1))
        k_controls = 0
    else:
        Xs = np.column_stack([np.ones(m.sum())] + [(sc == lv).astype(float) for lv in levels[1:]])
        k_controls = Xs.shape[1] - 1

    bY, *_ = np.linalg.lstsq(Xs, score[m], rcond=None)
    bA, *_ = np.linalg.lstsq(Xs, age[m],   rcond=None)
    residY = score[m] - Xs @ bY
    residA = age[m]   - Xs @ bA

    sY = np.std(residY, ddof=1); sA = np.std(residA, ddof=1)
    if sY == 0 or sA == 0:
        return np.nan, np.nan, np.nan

    r = np.corrcoef(residY, residA)[0,1]
    df = Xs.shape[0] - k_controls - 2
    if df <= 0:
        return r, np.nan, df
    t = r * np.sqrt(df / (1 - r*r)) if (1 - r*r) > 0 else np.nan
    p = 2 * (1 - stats.t.cdf(abs(t), df)) if np.isfinite(t) else np.nan
    return r, p, df


def pretty_dataset(label: str) -> str:
    return "ComBat-GAM" if "_HARM_" in label.upper() else "Unharmonized"


def pretty_wave(label: str) -> str:
    s = label.upper()
    if "_WAVEI_" in s:  return "ALSPAC MRI-I"
    if "_WAVEII_" in s: return "ALSPAC MRI-II"
    return "Pooled"


def panel_title(dataset: str, wave: str, k_top: int, pc=None):
    suffix = f"(K={k_top}, PC{pc})" if pc is not None else f"(K={k_top}, fused)"
    return f"{wave} -- {dataset} {suffix}"


def annot_str(rho, p, n):
    if rho is None or not np.isfinite(rho): return f"N={n}"
    p_txt = "<0.001" if (p is not None and np.isfinite(p) and p < 1e-3) else (f"{p:.3f}" if p is not None and np.isfinite(p) else "nan")
    return f"rho={rho:+.3f}, p={p_txt}, N={n}"


def dose_plot(pl, y, title, ylabel, out_svg, ylims=None, annot=None, annot_loc="upper left"):
    tmp = pd.DataFrame({'PLIKS': pl, 'val': y})
    tmp = tmp[np.isfinite(tmp['PLIKS']) & np.isfinite(tmp['val'])]
    if tmp.empty:
        return
    g = tmp.groupby('PLIKS')['val']
    means = g.mean(); ns = g.size(); sds = g.std(ddof=1); ses = sds/np.sqrt(ns)
    xs = np.array(sorted(means.index))
    plt.figure(figsize=(6,4.2))
    ax = plt.gca()
    ax.axhline(0, ls='--', lw=1)
    ax.errorbar(xs, means.loc[xs], yerr=1.96*ses.loc[xs], fmt='o-')
    ax.set_xticks(xs, [f'{int(x)} (n={ns.loc[x]})' for x in xs])
    ax.set_xlabel('PLIKS (0-3)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylims is not None:
        ax.set_ylim(ylims)
    if annot:
        loc_xy = {
            "upper left":  (0.02, 0.98, "left",  "top"),
            "upper right": (0.98, 0.98, "right", "top"),
            "lower left":  (0.02, 0.02, "left",  "bottom"),
            "lower right": (0.98, 0.02, "right", "bottom"),
        }[annot_loc]
        ax.text(loc_xy[0], loc_xy[1], annot,
                transform=ax.transAxes, ha=loc_xy[2], va=loc_xy[3],
                fontsize=9, bbox=dict(fc="white", alpha=0.8, ec="none"))
    plt.tight_layout(); plt.savefig(out_svg, dpi=600); plt.close()


def age_scatter_controls(age, score, mask_ctrl, title, out_svg):
    m = np.isfinite(age) & np.isfinite(score) & mask_ctrl
    if m.sum() < 10:
        return
    r, p = stats.spearmanr(age[m], score[m])
    plt.figure(figsize=(5.6,4.4))
    plt.scatter(age[m], score[m], s=12)
    b1, a1 = np.polyfit(age[m], score[m], 1)
    xs = np.linspace(np.nanmin(age[m]), np.nanmax(age[m]), 100)
    plt.plot(xs, a1 + b1*xs)
    plt.xlabel('Age (controls)'); plt.ylabel('PCA age-axis score (+older-like)')
    plt.title(f"{title}\nSpearman r={r:.2f}, p={fmt_p(p)}")
    plt.tight_layout(); plt.savefig(out_svg, dpi=600); plt.close()


def weights_barplot(rois, weights, title, out_svg, top=20):
    idx = np.argsort(np.abs(weights))[::-1][:top]
    plt.figure(figsize=(8, 0.35*top + 1.5))
    plt.barh(range(top), weights[idx])
    plt.yticks(range(top), [rois[i] for i in idx])
    plt.gca().invert_yaxis()
    plt.xlabel("Weight (+older-like)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_svg, dpi=600); plt.close()


def rho_barplot(rho_ctrl_raw, rho_ctrl_or, rho_all_raw, rho_all_or, out_svg):
    K = len(rho_ctrl_raw)
    x = np.arange(K)
    width = 0.22
    plt.figure(figsize=(max(6, K*0.25+3), 4.2))
    plt.bar(x - 1.5*width, np.abs(rho_ctrl_raw), width, label='Controls raw')
    plt.bar(x - 0.5*width, np.abs(rho_ctrl_or),  width, label='Controls oriented')
    plt.bar(x + 0.5*width, np.abs(rho_all_raw),  width, label='All rows raw')
    plt.bar(x + 1.5*width, np.abs(rho_all_or),   width, label='All rows oriented')
    plt.xlabel('PC index'); plt.ylabel('|Spearman(age, PC score)|')
    plt.title('Age-PC correlations (|rho|)')
    plt.legend()
    plt.tight_layout(); plt.savefig(out_svg, dpi=600); plt.close()


def align_waves_to_long(df1, df2, rois):
    keep1 = ['ID','Edad'] + rois
    keep2 = ['ID','Edad'] + rois

    sc1 = pick_scanner_col(df1)
    sc2 = pick_scanner_col(df2)
    if sc1 and sc1 not in keep1: keep1.append(sc1)
    if sc2 and sc2 not in keep2: keep2.append(sc2)

    pl1 = pick_pliks(df1)
    pl2 = pick_pliks(df2)
    if pl1: keep1 += pl1
    if pl2: keep2 += pl2

    a = df1[keep1].copy();  a['Time'] = 0
    b = df2[keep2].copy();  b['Time'] = 1
    M = a.merge(b, on='ID', suffixes=('_t1','_t2'))
    if M.empty:
        return pd.DataFrame(), pd.DataFrame()

    t1 = M[['ID']+[c for c in M.columns if c.endswith('_t1')]].copy()
    t1.columns = [c.replace('_t1','') for c in t1.columns]
    t1['Time'] = 0
    t2 = M[['ID']+[c for c in M.columns if c.endswith('_t2')]].copy()
    t2.columns = [c.replace('_t2','') for c in t2.columns]
    t2['Time'] = 1

    def add_pliks_columns(df,
                          col18_candidates=('PLIKS18TH_t1', 'pliks18TH'),
                          col30_candidates=('PLIKS30TH_t2', 'pliks30TH'),
                          out_comb='PLIKSc', out_pliks='PLIKS'):
        def pick_col(cands):
            for c in cands:
                if c in df.columns:
                    return c
            return None

        c18 = pick_col(col18_candidates)
        c30 = pick_col(col30_candidates)

        v18 = pd.to_numeric(df[c18], errors='coerce').clip(0, 3) if c18 else pd.Series(np.nan, index=df.index, dtype=float)
        v30 = pd.to_numeric(df[c30], errors='coerce').clip(0, 3) if c30 else pd.Series(np.nan, index=df.index, dtype=float)

        df[out_comb] = pd.concat([v18, v30], axis=1).max(axis=1, skipna=True)
        df[out_pliks] = v18
        return df

    t1 = add_pliks_columns(t1, pl1)
    t2 = add_pliks_columns(t2, pl2)

    L = pd.concat([t1, t2], axis=0, ignore_index=True)
    return M, L


def fit_pca_age_axis_on_long_controls(I_df, II_df, HROIs, label, k_top=1, max_pcs=MAX_PCS, outdir="."):
    ensure_dir(outdir)
    M, L = align_waves_to_long(I_df, II_df, HROIs)
    if L.empty:
        raise RuntimeError("No paired IDs to train the age axis.")

    X   = L[HROIs].apply(pd.to_numeric, errors='coerce').values
    age = pd.to_numeric(L['Edad'], errors='coerce').values
    pl  = pd.to_numeric(L['PLIKSc'], errors='coerce').astype(float).values

    gb = pd.Series(pl, index=L.index).groupby(L['ID'])
    ctrl_by_id = gb.apply(lambda s: (np.all(np.isfinite(s)) and np.all(s==0) and s.size>=2))
    mask_ctrl = L['ID'].map(ctrl_by_id).astype(bool).values

    if mask_ctrl.sum() < 10:
        raise RuntimeError(f"Need >=10 control rows to fit scaler/PCA robustly (have {mask_ctrl.sum()}).")

    X_imp, scaler, med = z_impute_controls(X, mask_ctrl)
    Xc_ctrl = scaler.transform(X_imp[mask_ctrl, :])

    n_ctrl_rows = Xc_ctrl.shape[0]
    n_feat = Xc_ctrl.shape[1]
    n_comp = int(min(max_pcs, n_feat, max(1, n_ctrl_rows-1)))
    pca = PCA(n_components=n_comp, svd_solver='full', random_state=0).fit(Xc_ctrl)

    scores_all_raw = pca.transform(scaler.transform(X_imp))

    rho_ctrl_raw, p_ctrl_raw = [], []
    rho_all_raw,  p_all_raw  = [], []
    for k in range(n_comp):
        s = scores_all_raw[:,k]
        m_ctrl = np.isfinite(s) & np.isfinite(age) & mask_ctrl
        m_all  = np.isfinite(s) & np.isfinite(age)
        r1,p1 = (stats.spearmanr(age[m_ctrl], s[m_ctrl]) if m_ctrl.sum() >= 20 else (np.nan, np.nan))
        r2,p2 = (stats.spearmanr(age[m_all],  s[m_all])  if m_all.sum()  >= 30 else (np.nan, np.nan))
        rho_ctrl_raw.append(r1); p_ctrl_raw.append(p1)
        rho_all_raw.append(r2);  p_all_raw.append(p2)
    rho_ctrl_raw = np.array(rho_ctrl_raw); p_ctrl_raw = np.array(p_ctrl_raw)
    rho_all_raw  = np.array(rho_all_raw);  p_all_raw  = np.array(p_all_raw)

    sgn = np.sign(np.nan_to_num(rho_ctrl_raw, nan=0.0)); sgn[sgn==0]=1.0
    scores_all = scores_all_raw * sgn[None,:]
    comps_oriented = (pca.components_.T * sgn).T

    rho_ctrl_or, p_ctrl_or = [], []
    rho_all_or,  p_all_or  = [], []
    for k in range(n_comp):
        s = scores_all[:,k]
        m_ctrl = np.isfinite(s) & np.isfinite(age) & mask_ctrl
        m_all  = np.isfinite(s) & np.isfinite(age)
        r1,p1 = (stats.spearmanr(age[m_ctrl], s[m_ctrl]) if m_ctrl.sum() >= 20 else (np.nan, np.nan))
        r2,p2 = (stats.spearmanr(age[m_all],  s[m_all])  if m_all.sum()  >= 30 else (np.nan, np.nan))
        rho_ctrl_or.append(r1); p_ctrl_or.append(p1)
        rho_all_or.append(r2);  p_all_or.append(p2)
    rho_ctrl_or = np.array(rho_ctrl_or); p_ctrl_or = np.array(p_ctrl_or)
    rho_all_or  = np.array(rho_all_or);  p_all_or  = np.array(p_all_or)

    q_ctrl_or = benjamini_hochberg(np.nan_to_num(p_ctrl_or, nan=1.0))

    sc_col = pick_scanner_col(L)
    scanner_labels = L[sc_col].astype(str).values if sc_col else np.array(['NA']*len(L))

    R2_age_all, R2_full_all, dR2_all = [], [], []
    F_all, pblk_all, df1_all, df2_all = [], [], [], []
    pr_all, pp_all = [], []

    for k in range(n_comp):
        s = scores_all[:, k]
        info = scanner_block_stats(score=s, age=age, scanner_labels=scanner_labels, mask_ctrl=mask_ctrl)
        R2_age_all.append(info['R2_age'])
        R2_full_all.append(info['R2_full'])
        dR2_all.append(info['deltaR2'])
        F_all.append(info['F'])
        pblk_all.append(info['p_block'])
        df1_all.append(info['df1'])
        df2_all.append(info['df2'])

        r_part, p_part, _ = partial_corr_age_given_scanner(s, age, scanner_labels, mask_ctrl)
        pr_all.append(r_part); pp_all.append(p_part)

    train_stats = pd.DataFrame({
        'PC': np.arange(n_comp, dtype=int),
        'rho_age_ctrl_raw': rho_ctrl_raw, 'p_age_ctrl_raw': p_ctrl_raw,
        'rho_age_ctrl_oriented': rho_ctrl_or, 'p_age_ctrl_oriented': p_ctrl_or, 'q_age_ctrl_oriented': q_ctrl_or,
        'rho_age_all_raw': rho_all_raw, 'p_age_all_raw': p_all_raw,
        'rho_age_all_oriented': rho_all_or, 'p_age_all_oriented': p_all_or,
        'explained_var_ratio': pca.explained_variance_ratio_,
        'R2_age_ctrl': R2_age_all,
        'R2_age_plus_scanner_ctrl': R2_full_all,
        'deltaR2_scanner_ctrl': dR2_all,
        'F_scanner_block_ctrl': F_all,
        'df1_scanner_block_ctrl': df1_all,
        'df2_scanner_block_ctrl': df2_all,
        'p_scanner_block_ctrl': pblk_all,
        'partial_r_age_given_scanner_ctrl': pr_all,
        'partial_p_age_given_scanner_ctrl': pp_all,
    })
    train_stats.to_csv(os.path.join(outdir, f"partA_PCA_training_stats_{label}.csv"), index=False)
    rho_barplot(rho_ctrl_raw, rho_ctrl_or, rho_all_raw, rho_all_or,
                os.path.join(outdir, f"partA_PCA_age_rhos_{label}.svg"))

    idx_sorted = np.argsort(-np.nan_to_num(np.abs(rho_ctrl_or), nan=0.0))
    sel = idx_sorted[:min(k_top, n_comp)]

    selected_rows = []
    b_age_per_pc = []
    for rank, k in enumerate(sel):
        resid_k, b_k = age_adjust_residuals(scores_all[:, k], age, mask_ctrl)
        b_age_per_pc.append(b_k)
        selected_rows.append({
            'rank': int(rank+1),
            'PC': int(k),
            'rho_age_ctrl_oriented': float(rho_ctrl_or[k]),
            'p_age_ctrl_oriented': float(p_ctrl_or[k]) if np.isfinite(p_ctrl_or[k]) else np.nan,
            'rho_age_all_oriented': float(rho_all_or[k]),
            'p_age_all_oriented': float(p_all_or[k]) if np.isfinite(p_all_or[k]) else np.nan,
            'explained_var_ratio': float(pca.explained_variance_ratio_[k]),
        })
    b_age_per_pc = np.vstack(b_age_per_pc) if len(b_age_per_pc)>0 else np.zeros((0,3))
    sel_df = pd.DataFrame(selected_rows)
    sel_df.to_csv(os.path.join(outdir, f"partA_PCA_SELECTEDPC_age_rhos_{label}_K{k_top}.csv"), index=False)

    print(f"\n[Training] {label} (K={k_top}) - Age correlations for selected PCs:")
    for row in selected_rows:
        pc = row['PC']
        r2a  = train_stats.loc[train_stats.PC==pc, 'R2_age_ctrl'].item()
        r2f  = train_stats.loc[train_stats.PC==pc, 'R2_age_plus_scanner_ctrl'].item()
        dr2  = train_stats.loc[train_stats.PC==pc, 'deltaR2_scanner_ctrl'].item()
        Fblk = train_stats.loc[train_stats.PC==pc, 'F_scanner_block_ctrl'].item()
        df1  = train_stats.loc[train_stats.PC==pc, 'df1_scanner_block_ctrl'].item()
        df2  = train_stats.loc[train_stats.PC==pc, 'df2_scanner_block_ctrl'].item()
        pblk = train_stats.loc[train_stats.PC==pc, 'p_scanner_block_ctrl'].item()
        prt  = train_stats.loc[train_stats.PC==pc, 'partial_r_age_given_scanner_ctrl'].item()
        ppt  = train_stats.loc[train_stats.PC==pc, 'partial_p_age_given_scanner_ctrl'].item()

        print(f"  rank {row['rank']:>2} | PC {pc:>2} | "
              f"rho_ctrl={row['rho_age_ctrl_oriented']:+.3f} (p={fmt_p(row['p_age_ctrl_oriented'])}) | "
              f"rho_all={row['rho_age_all_oriented']:+.3f} (p={fmt_p(row['p_age_all_oriented'])}) | "
              f"EVR={row['explained_var_ratio']:.3f}")
        print(f"[Scanner vs Age | controls | {label}] PC{pc}: "
              f"R2(Age)= {r2a:.3f} -> R2(+Scanner)= {r2f:.3f} (deltaR2={dr2:+.3f}); "
              f"F={np.nan if not np.isfinite(Fblk) else float(Fblk):.2f} (df={int(df1) if np.isfinite(df1) else 0},{int(df2) if np.isfinite(df2) else 0}), "
              f"p_block={fmt_p(pblk)}; partial r(Age|Scanner)= {np.nan if not np.isfinite(prt) else float(prt):.3f} (p={fmt_p(ppt)})")

    w_fuse = np.abs(rho_ctrl_or[sel])
    w_fuse = w_fuse / (w_fuse.sum() if np.isfinite(w_fuse).sum()>0 else len(sel))
    fused = np.sum(scores_all[:, sel] * w_fuse[None, :], axis=1)
    resid_fused, b_age = age_adjust_residuals(fused, age, mask_ctrl)

    subj = pd.DataFrame({'ID': L['ID'].values, 'Time': L['Time'].values, 'Edad': age, 'PLIKS': pl,
                         'fused_score': fused, 'fused_resid_ageAdj': resid_fused})
    for i, k in enumerate(sel):
        r_k, _ = age_adjust_residuals(scores_all[:, k], age, mask_ctrl)
        subj[f'PC{k}_score_oriented'] = scores_all[:, k]
        subj[f'PC{k}_resid_ageAdj_TRAINONLY'] = r_k
    subj.to_csv(os.path.join(outdir, f"partA_PCA_subjects_long_controlsTraining_{label}_K{k_top}.csv"), index=False)

    age_scatter_controls(age, fused, mask_ctrl,
                         f"{label}: PCA fused age-axis vs age (controls; longitudinal rows)",
                         os.path.join(outdir, f"partA_PCA_fused_vs_age_controls_{label}_K{k_top}.svg"))
    weights_barplot(HROIs,
                    np.sum((comps_oriented[sel,:].T * w_fuse)[...,None], axis=1).ravel(),
                    f"{label}: PCA age-axis weights (top |w|; fused, K={k_top})",
                    os.path.join(outdir, f"partA_PCA_fused_weights_top20_{label}_K{k_top}.svg"), top=20)

    return {
        'HROIs': HROIs,
        'med': med, 'scaler': scaler, 'pca': pca,
        'sgn': sgn, 'sel': sel.astype(int), 'w_fuse': w_fuse,
        'b_age': b_age,
        'b_age_per_pc': b_age_per_pc,
        'rho_ctrl_or': rho_ctrl_or, 'rho_all_or': rho_all_or,
        'pair_rows': len(L), 'pair_ids': len(M),
        'n_ctrl_rows': int(mask_ctrl.sum())
    }


def project_age_axis(df, model, label, outdir):
    ensure_dir(outdir)
    X = df[model['HROIs']].apply(pd.to_numeric, errors='coerce').values
    age = pd.to_numeric(df['Edad'], errors='coerce').values
    pl_col = 'pliks18TH'
    pl_ser = pd.to_numeric(df.get(pl_col, np.nan), errors='coerce').astype(float)
    pl_ser = pl_ser.clip(lower=0, upper=3)

    X_imp = np.where(np.isnan(X), model['med'][None,:], X)
    scores = model['pca'].transform(model['scaler'].transform(X_imp)) * model['sgn'][None,:]

    dataset = pretty_dataset(label)
    wave    = pretty_wave(label)
    ylims = YLIMS_BY_DATASET.get(dataset)
    if ylims is not None:
        factor = 1.60
        ylims = tuple(factor * x for x in ylims)

    fused  = np.sum(scores[:, model['sel']] * model['w_fuse'][None,:], axis=1)
    resid = np.full_like(fused, np.nan, dtype=float)
    m = np.isfinite(fused) & np.isfinite(age)
    if np.all(np.isfinite(model['b_age'])):
        Xa = np.column_stack([np.ones(m.sum()), age[m], age[m]**2])
        resid[m] = fused[m] - Xa @ model['b_age']
    else:
        resid[m] = fused[m] - np.nanmean(fused[m])

    out = df.copy()
    out['pca_age_fused'] = fused
    out['pca_age_resid'] = resid
    out['PLIKS'] = pl_ser

    pc_stats = []
    for i, k in enumerate(model['sel']):
        s_k = scores[:, k]
        r_k = np.full_like(s_k, np.nan, dtype=float)
        m_k = np.isfinite(s_k) & np.isfinite(age)
        if model['b_age_per_pc'].shape[0] > i and np.all(np.isfinite(model['b_age_per_pc'][i])):
            Xa = np.column_stack([np.ones(m_k.sum()), age[m_k], age[m_k]**2])
            r_k[m_k] = s_k[m_k] - Xa @ model['b_age_per_pc'][i]
        else:
            r_k[m_k] = s_k[m_k] - np.nanmean(s_k[m_k])
        out[f'PC{k}_score_oriented'] = s_k
        out[f'PC{k}_resid_ageAdj']   = r_k

        m2 = np.isfinite(out['PLIKS'].values) & np.isfinite(r_k)
        if m2.sum() >= 30:
            rho_pc, p_pc = stats.spearmanr(out['PLIKS'].values[m2], r_k[m2])
            print(f"[{label}] PC{k}: Spearman(PLIKS, PC_resid) = {rho_pc:+.3f}, p={fmt_p(p_pc)} | N={m2.sum()}")
        else:
            rho_pc, p_pc = (np.nan, np.nan)
            print(f"[{label}] PC{k}: Not enough data for correlation (N={m2.sum()}).")

        dose_plot(
            out['PLIKS'].values, r_k,
            title=panel_title(dataset, wave, k_top=len(model['sel']), pc=k),
            ylabel=f"PC{k} residual (older-like)",
            out_svg=os.path.join(outdir, f"partA_PCA_dose_{label}_PC{k}.svg"),
            ylims=ylims,
            annot=annot_str(rho_pc, p_pc, int(m2.sum())),
            annot_loc="upper left",
        )

        pc_stats.append({'PC': int(k), 'rho_PLIKS_vs_PCresid': float(rho_pc), 'p': float(p_pc), 'N': int(m2.sum())})
    if pc_stats:
        pd.DataFrame(pc_stats).to_csv(os.path.join(outdir, f"partA_PCA_perPC_vs_PLIKS_{label}.csv"), index=False)

    m2 = np.isfinite(out['PLIKS'].values) & np.isfinite(resid)
    if m2.sum() >= 30:
        rho, p = stats.spearmanr(out['PLIKS'].values[m2], resid[m2])
        print(f"[{label}] Spearman(PLIKS, fused_age_resid) = {rho:+.3f}, p={fmt_p(p)} | N={m2.sum()}")
    else:
        rho, p = (np.nan, np.nan)
        print(f"[{label}] Not enough data for correlation (N={m2.sum()}).")

    dose_plot(
        out['PLIKS'].values, resid,
        title=panel_title(dataset, wave, k_top=len(model['sel']), pc=None),
        ylabel="Fused age-axis residual (older-like)",
        out_svg=os.path.join(outdir, f"partA_PCA_dose_{label}.svg"),
        ylims=ylims,
        annot=annot_str(rho, p, int(m2.sum())),
        annot_loc="upper left",
    )

    out.to_csv(os.path.join(outdir, f"partA_age_axis_projection_{label}.csv"), index=False)
    return out


def run_partA_pca_long_controls(I_df, II_df, HROIs, label, outdir, k_top=1):
    ensure_dir(outdir)
    model = fit_pca_age_axis_on_long_controls(I_df, II_df, HROIs, label=label, k_top=k_top, max_pcs=MAX_PCS, outdir=outdir)
    print(f"\n=== Part A (PCA@controls) - {label} (K={k_top}) ===")
    print(f"rows(paired long)={model['pair_rows']} | pairs={model['pair_ids']} | controls(rows)={model['n_ctrl_rows']} | HROIs={len(HROIs)}")
    print(f"Top-K PCs: {list(map(int, model['sel']))} (per-PC analyses enabled)")

    _ = project_age_axis(I_df,  model, label=f"{label}_WaveI_FULL_K{k_top}",  outdir=outdir)
    _ = project_age_axis(II_df, model, label=f"{label}_WaveII_FULL_K{k_top}", outdir=outdir)

    return {'label': label, 'rows': model['pair_rows'], 'pairs': model['pair_ids'],
            'n_ctrl_rows': model['n_ctrl_rows'], 'K': len(model['sel'])}


def compute_apc(X1, X2, dY, mode="standard"):
    if mode == "symmetric":
        num = 200.0 * (X2 - X1)
        den = (np.abs(X2) + np.abs(X1))
        APC = num / den
    else:
        eps = 1e-8
        den = np.where(np.abs(X1) < eps, np.nan, X1)
        APC = 100.0 * (X2 - X1) / den
    return APC / dY[:, None]


def run_partB(df1, df2, HROIs, label, outdir):
    ensure_dir(outdir)
    keep2 = ['ID','Edad'] + HROIs
    for cov in ['sexo(M=1;F=0)','eTIV']:
        if cov in df2.columns: keep2.append(cov)
    pl_col = pick_pliks(df2)
    if pl_col: keep2.append(pl_col[0])

    a = df1[['ID','Edad'] + HROIs].copy()
    b = df2[keep2].copy()
    M = a.merge(b, on='ID', suffixes=('_t1','_t2'))
    if M.empty:
        print(f"\n=== Part B - {label} ===\nNo overlapping IDs.")
        return {'label': label, 'N': 0, 'n_ok': 0, 'rho': np.nan, 'p_rho': np.nan, 'beta': np.nan, 'se': np.nan}

    age1 = pd.to_numeric(M['Edad_t1'], errors='coerce').values
    age2 = pd.to_numeric(M['Edad_t2'], errors='coerce').values
    dY = age2 - age1

    if pl_col and pl_col[0] + '_t2' in M.columns:
        M = M.rename(columns={pl_col[0] + '_t2': 'PLIKS_t2'})
    elif pl_col and pl_col[0] in M.columns:
        M = M.rename(columns={pl_col[0]: 'PLIKS_t2'})
    pl2 = pd.to_numeric(M.get('PLIKS_t2', pd.Series(np.nan, index=M.index, dtype=float)), errors='coerce').astype(float).values
    pl2[(pl2<0)|(pl2>3)] = np.nan

    X1 = M[[c+'_t1' for c in HROIs]].apply(pd.to_numeric, errors='coerce').values
    X2 = M[[c+'_t2' for c in HROIs]].apply(pd.to_numeric, errors='coerce').values

    APC_yr = compute_apc(X1, X2, dY, mode=APC_MODE)
    mean_abs_apc = np.nanmean(np.abs(APC_yr), axis=1)

    ok = np.isfinite(mean_abs_apc) & np.isfinite(pl2) & np.isfinite(dY) & (dY > 0.1)
    n_ok = int(ok.sum())

    rho, p_rho = (stats.spearmanr(pl2[ok], mean_abs_apc[ok]) if n_ok>=30 else (np.nan, np.nan))

    cov_list = []
    if 'sexo(M=1;F=0)_t2' in M.columns: cov_list.append(pd.to_numeric(M['sexo(M=1;F=0)_t2'], errors='coerce').values)
    if 'eTIV_t2' in M.columns:           cov_list.append(pd.to_numeric(M['eTIV_t2'], errors='coerce').values)
    cov_arr = np.column_stack([c for c in cov_list]) if cov_list else None

    if n_ok>0:
        if cov_arr is not None:
            Xols = np.column_stack([np.ones(n_ok), pl2[ok], cov_arr[ok,:]])
        else:
            Xols = np.column_stack([np.ones(n_ok), pl2[ok]])
    else:
        Xols = np.ones((0,2))

    try:
        inv = np.linalg.inv(Xols.T @ Xols) if n_ok>0 else None
        beta = (inv @ Xols.T @ mean_abs_apc[ok])[1] if n_ok>0 else np.nan
        res = mean_abs_apc[ok] - Xols @ (inv @ Xols.T @ mean_abs_apc[ok]) if n_ok>0 else np.array([])
        dof = max(n_ok - Xols.shape[1], 1) if n_ok>0 else 1
        sigma2 = float(res.T @ res / dof) if n_ok>0 else np.nan
        se = np.sqrt(np.diag(inv) * sigma2)[1] if n_ok>0 else np.nan
    except np.linalg.LinAlgError:
        beta, se = np.nan, np.nan

    out = pd.DataFrame({
        'ID': M['ID'].values,
        'Edad_t1': age1, 'Edad_t2': age2, 'DeltaYears': dY,
        'PLIKS_t2': pl2,
        'MeanAbsAPC_perYear': mean_abs_apc
    })
    out.to_csv(os.path.join(outdir, f'partB_subject_APC_{label}.csv'), index=False)

    comp = (out.assign(PLIKS_grp=out['PLIKS_t2'])
                .groupby('PLIKS_grp')[['DeltaYears','MeanAbsAPC_perYear']]
                .agg(['count','mean','std']))
    comp.to_csv(os.path.join(outdir, f'partB_deltaYears_comparability_{label}.csv'))

    tmp = out[ok]
    if not tmp.empty:
        g = tmp.groupby('PLIKS_t2')['MeanAbsAPC_perYear']
        means = g.mean(); ns = g.size(); sds = g.std(ddof=1); ses = sds/np.sqrt(ns)
        xs = np.array(sorted(means.index))
        plt.figure(figsize=(6.0,4.2))
        ax = plt.gca()
        ax.errorbar(xs, means.loc[xs], yerr=1.96*ses.loc[xs], fmt='o-')
        ax.set_xticks(xs, [f'{int(x)} (n={ns.loc[x]})' for x in xs])
        ax.set_xlabel('PLIKS (0-3)')
        ax.set_ylabel('Mean |APC| (% per year) across ROIs')
        ax.set_title(f'{label}: Longitudinal mean |APC| by PLIKS')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'partB_APC_by_PLIKS_{label}.svg'), dpi=600); plt.close()

    print(f"\n=== Part B - {label} ===")
    print(f"HROIs={len(HROIs)} | Paired IDs={len(out)} | usable (DeltaYears>0.1 & finite)={n_ok}")
    print(f"Spearman(PLIKS_t2, mean|APC|/yr): rho={rho:.3f}, p={fmt_p(p_rho)}")
    print(f"OLS mean|APC|/yr ~ PLIKS: beta={beta:.4f} +/- {se:.4f}")
    print(f"[Saved] subject APCs, DeltaYears comparability, plot, stats -> {outdir}")
    return {'label': label, 'N': len(out), 'n_ok': n_ok, 'rho': float(rho), 'p_rho': float(p_rho),
            'beta': float(beta), 'se': float(se)}


if __name__ == "__main__":
    I_h = pd.read_csv(I_HARMO)
    II_h = pd.read_csv(II_HARMO)

    if os.path.exists(I_UNHARM) and os.path.exists(II_UNHARM):
        I_u = pd.read_csv(I_UNHARM)
        II_u = pd.read_csv(II_UNHARM)
        have_unharm = True
    else:
        print(f"[note] Unharmonized inputs not found at '{I_UNHARM}' / '{II_UNHARM}'; "
              f"skipping UNHARM analyses.")
        I_u = None
        II_u = None
        have_unharm = False

    for df in (I_h, II_h, I_u, II_u):
        if df is None: continue
        df['ID'] = df['ID'].astype(str).map(canon_id)

    ids_to_remove = ['SUBJ_00432', 'SUBJ_00557', 'SUBJ_00579',
                     'SUBJ_00351', 'SUBJ_00501']
    I_h = I_h[~I_h['ID'].isin(ids_to_remove)]
    if I_u is not None:
        I_u = I_u[~I_u['ID'].isin(ids_to_remove)]

    I_h, II_h, I_u, II_u = backfill_pliks_columns(I_h, II_h, I_u, II_u)

    HROIs_I  = rois_from_harmonized(I_h)
    HROIs_II = rois_from_harmonized(II_h)

    if have_unharm:
        HROIs_I_UNH  = [c for c in HROIs_I  if c in I_u.columns  and pd.api.types.is_numeric_dtype(I_u[c])  and not c.lower().startswith('pliks')]
        HROIs_II_UNH = [c for c in HROIs_II if c in II_u.columns and pd.api.types.is_numeric_dtype(II_u[c]) and not c.lower().startswith('pliks')]
        HROIs_B_UNH  = sorted([c for c in (set(HROIs_I_UNH) & set(HROIs_II_UNH)) if not c.lower().startswith('pliks')])
    else:
        HROIs_B_UNH = []
    HROIs_B_HARM = sorted([c for c in (set(HROIs_I) & set(HROIs_II)) if not c.lower().startswith('pliks')])

    summaries_A = []
    summaries_B = []

    print("\n########## PART A (PCA age-axis; strict longitudinal controls-only) ##########")
    for K in K_LIST:
        summaries_A.append(run_partA_pca_long_controls(I_h, II_h, HROIs_B_HARM,
                                                       label=f"Cardiff_Long_HARM_K{K}",
                                                       outdir=os.path.join(OUTDIR, "HARM"),
                                                       k_top=K))
        if have_unharm:
            summaries_A.append(run_partA_pca_long_controls(I_u, II_u, HROIs_B_UNH,
                                                           label=f"Cardiff_Long_UNHARM_K{K}",
                                                           outdir=os.path.join(OUTDIR, "UNHARM"),
                                                           k_top=K))

    print("\n########## PART B: Longitudinal I->II (APC) ##########")
    summaries_B.append(run_partB(I_h,  II_h,  HROIs_B_HARM, label="ALSPAC MRI-I to II Harmonized",  outdir=os.path.join(OUTDIR, "HARM")))
    if have_unharm:
        summaries_B.append(run_partB(I_u,  II_u,  HROIs_B_UNH,  label="ALSPAC MRI-I to II Unharmonized", outdir=os.path.join(OUTDIR, "UNHARM")))

    print("\n==================== FINAL RECAP ====================")
    print("Part A (PCA age-axis; strict longitudinal controls-only; per-PC + fused):")
    for s in summaries_A:
        if s is None: continue
        print(f" - {s['label']}: rows={s['rows']}, pairs={s['pairs']}, controls(rows)={s['n_ctrl_rows']}, K={s['K']}")
    print("Part B (APC longitudinal):")
    for s in summaries_B:
        if s is None: continue
        print(f" - {s['label']}: usable N={s['n_ok']}, rho={s['rho']:.3f}, p={fmt_p(s['p_rho'])}, "
              f"beta={s['beta']:.4f} +/- {s['se']:.4f}")

    print(f"\nDone. See {OUTDIR}/HARM and {OUTDIR}/UNHARM for outputs.")
