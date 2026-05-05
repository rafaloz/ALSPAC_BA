"""
01_harmonize.py
================

ComBat-GAM harmonization of the application/test cohorts against the
multi-site training reference. The training set itself is harmonized
upstream (single reference scale) and is loaded here as an already-
harmonized CSV.

Workflow
--------
1. Load the training reference (already harmonized).
2. For each test cohort:
   - drop 31 unreliable FastSurfer columns (Choroid Plexus, hypointensities,
     vessels, surface holes, mask/total-volume summaries — see Table S3 in
     the supplement; note the supplement lists 32 incl. eTIV but in the code
     eTIV is kept as a feature and also used as a ComBat-GAM covariate);
   - for ALSPAC waves: split by PE status, fit on PE-negative controls, apply
     to PE-positive participants (avoids label leakage);
   - for AgeRisk: single-batch fit + apply.
3. Save the harmonized CSVs into ../data/.

Run from `src/`. ComBat-GAM model pickles are dropped in CWD.
"""

import os
import pandas as pd

from utils.harmonize_utils import learn_harmonization, apply_harmonization


# 31 FastSurfer columns dropped before ComBat-GAM
PROBLEMATIC_FEATURES = [
    # Choroid Plexus / Ventricles
    "VentricleChoroidVol",
    "Volume_mm3_Left-choroid-plexus",
    "Volume_mm3_Right-choroid-plexus",
    # Hypointensities / Vessels
    "Volume_mm3_WM-hypointensities",
    "Volume_mm3_Right-WM-hypointensities",
    "Volume_mm3_Left-WM-hypointensities",
    "Volume_mm3_non-WM-hypointensities",
    "Volume_mm3_Right-non-WM-hypointensities",
    "Volume_mm3_Left-non-WM-hypointensities",
    "Volume_mm3_Left-vessel",
    "Volume_mm3_Right-vessel",
    # Other small / unstable structures
    "Volume_mm3_Optic-Chiasm",
    "Volume_mm3_5th-Ventricle",
    # Surface-mesh quality counters
    "lhSurfaceHoles",
    "rhSurfaceHoles",
    "SurfaceHoles",
    # Whole-brain summary volumes (collinear with covariates)
    "MaskVol",
    "MaskVol-to-eTIV",
    "BrainSegVol",
    "BrainSegVolNotVent",
    "BrainSegVolNotVentSurf",
    "BrainSegVol-to-eTIV",
    "SupraTentorialVol",
    "SupraTentorialVolNotVent",
    "SupraTentorialVolNotVentVox",
    "CortexVol",
    "TotalGrayVol",
    "CerebralWhiteMatterVol",
    "lhCerebralWhiteMatterVol",
    "rhCerebralWhiteMatterVol",
    "SubCortGrayVol",
]
assert len(PROBLEMATIC_FEATURES) == 31, "Expected 31 columns to drop"


# Paths
DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAIN_REFERENCE_CSV = os.path.join(DATA_DIR, "training_data_harmonized.csv")
AGERISK_RAW_CSV     = os.path.join(DATA_DIR, "AgeRisk_noHarmo.csv")
ALSPAC_20_RAW_CSV   = os.path.join(DATA_DIR, "ALSPAC_20_MRI_I_raw.csv")
ALSPAC_30_RAW_CSV   = os.path.join(DATA_DIR, "ALSPAC_30_MRI_II_raw.csv")

AGERISK_HARMO_OUT     = os.path.join(DATA_DIR, "AgeRisk_harmonized.csv")
ALSPAC_20_HARMO_OUT   = os.path.join(DATA_DIR, "ALSPAC_20_MRI_I_harmonized.csv")
ALSPAC_30_HARMO_OUT   = os.path.join(DATA_DIR, "ALSPAC_30_MRI_II_harmonized.csv")

MODEL_SAVE_DIR = "."


def drop_problematic(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=PROBLEMATIC_FEATURES, errors="ignore")


def harmonize_pe_cohort(train_reference: pd.DataFrame,
                        cohort_raw: pd.DataFrame,
                        harmo_name: str) -> pd.DataFrame:
    """Fit on PE-negative controls, apply to PE-positive participants."""
    controls = cohort_raw[cohort_raw["pliks18TH"] == 0].copy()
    pe_pos   = cohort_raw[cohort_raw["pliks18TH"] != 0].copy()

    learned, _ = learn_harmonization(train_reference, controls, harmo_name,
                                     model_save_dir=MODEL_SAVE_DIR)
    controls_harmo = learned[learned["Escaner"] != "zarmonization_1"]

    pe_pos_harmo = apply_harmonization(pe_pos, train_reference,
                                       ref_level=1,
                                       save_dir=MODEL_SAVE_DIR,
                                       name=harmo_name)
    return pd.concat([controls_harmo, pe_pos_harmo], axis=0)


# 1. Training reference (must be supplied by the user — pre-harmonized)
X_train = pd.read_csv(TRAIN_REFERENCE_CSV)


# 2. AgeRisk external test cohort
agerisk_raw = pd.read_csv(AGERISK_RAW_CSV)
agerisk_raw = drop_problematic(agerisk_raw)
agerisk_harmo, _ = learn_harmonization(X_train, agerisk_raw,
                                       "Armo_AgeRisk",
                                       model_save_dir=MODEL_SAVE_DIR)
agerisk_harmo = agerisk_harmo[agerisk_harmo["Escaner"] != "zarmonization_1"]
agerisk_harmo.to_csv(AGERISK_HARMO_OUT, index=False)


# 3. ALSPAC-20 MRI-I (Cardiff wave I)
alspac_20_raw = pd.read_csv(ALSPAC_20_RAW_CSV)
# 5 ALSPAC-20 scans excluded for quality (motion / segmentation failures).
# Listed as the obfuscated SUBJ_NNNNN IDs that ship in this repo. If you
# rerun on un-obfuscated raw data, replace these with the original ALSPAC
# clinic IDs (look them up via tools/id_mapping.csv if you have it locally).
ids_to_remove = ["SUBJ_00432", "SUBJ_00557", "SUBJ_00579",
                 "SUBJ_00351", "SUBJ_00501"]
alspac_20_raw = alspac_20_raw[~alspac_20_raw["ID"].isin(ids_to_remove)]
alspac_20_raw = drop_problematic(alspac_20_raw)
alspac_20_harmo = harmonize_pe_cohort(X_train, alspac_20_raw,
                                      "Armo_ALSPAC_20_MRI_I")
alspac_20_harmo.to_csv(ALSPAC_20_HARMO_OUT, index=False)


# 4. ALSPAC-30 MRI-II (Cardiff wave II)
alspac_30_raw = pd.read_csv(ALSPAC_30_RAW_CSV)
alspac_30_raw = drop_problematic(alspac_30_raw)
alspac_30_harmo = harmonize_pe_cohort(X_train, alspac_30_raw,
                                      "Armo_ALSPAC_30_MRI_II")
alspac_30_harmo.to_csv(ALSPAC_30_HARMO_OUT, index=False)


print("Harmonization complete. Outputs written to:", DATA_DIR)
