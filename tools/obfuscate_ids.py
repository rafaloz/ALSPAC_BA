"""
tools/obfuscate_ids.py

One-shot helper to anonymise subject IDs across the source CSVs.
Builds a single map (real_ID -> SUBJ_NNNNN), applies it consistently
to every column that holds a participant ID, and writes the cleaned
files into ../data/.

Run once from the repo root:
    python tools/obfuscate_ids.py
"""

import os
import re
import pandas as pd

ALSPAC_REVIEW = "/home/rafa/PycharmProjects/ALSPAC_review"
RAW_FS_DIR    = "/home/rafa/PycharmProjects/JoinData_FastSurfer_V2/Scripts_Join_data/Datos_per_DB"
DST_DIR       = "/home/rafa/PycharmProjects/ALSPAC_BA/data"
MAPPING_DST   = "/home/rafa/PycharmProjects/ALSPAC_BA/tools/id_mapping.csv"

# (source path, dest path, list-of-id-cols)
# Note: lmm_input_long.csv and lmm_input_long_pairs.csv are produced by
# src/07_export_LMM_input.py from the prediction CSVs — NOT obfuscated here.
SOURCES = [
    (f"{ALSPAC_REVIEW}/DatosNoEB/datos_morfo_Harmo_7_50_FF_noEB_2.csv",
     f"{DST_DIR}/training_data_harmonized.csv", ["ID"]),
    (f"{ALSPAC_REVIEW}/DatosNoEB/AgeRisk_harmo_7_50_FF_noEB_2.csv",
     f"{DST_DIR}/AgeRisk_harmonized.csv", ["ID"]),
    (f"{ALSPAC_REVIEW}/DatosNoEB/X_test_Cardiff_I_harmo_7_50_FF_noEB_2.csv",
     f"{DST_DIR}/ALSPAC_20_MRI_I_harmonized.csv", ["ID"]),
    (f"{ALSPAC_REVIEW}/DatosNoEB/X_test_Cardiff_II_harmo_7_50_FF_noEB_2.csv",
     f"{DST_DIR}/ALSPAC_30_MRI_II_harmonized.csv", ["ID"]),
    (f"{ALSPAC_REVIEW}/DatosReview/ALSPAC_I_depression.csv",
     f"{DST_DIR}/ALSPAC_I_depression.csv", ["id"]),
    (f"{ALSPAC_REVIEW}/DatosReview/ALSPAC_II_depression.csv",
     f"{DST_DIR}/ALSPAC_II_depression.csv", ["ID", "id"]),
    (f"{ALSPAC_REVIEW}/DatosReview/QC_ALSPAC/QC_fsqc_I_adq/fsqc-results.csv",
     f"{DST_DIR}/QC_fsqc_I_adq_fsqc-results.csv", ["subject"]),
    (f"{ALSPAC_REVIEW}/DatosReview/QC_ALSPAC/QC_fsqc_II_adq/fsqc-results.csv",
     f"{DST_DIR}/QC_fsqc_II_adq_fsqc-results.csv", ["subject"]),
    # Raw (un-harmonized) FastSurfer features — needed by 10_model_free_validation
    (f"{RAW_FS_DIR}/CARDIFF_PE_I_FastSurfer_V2_data.csv",
     f"{DST_DIR}/ALSPAC_20_MRI_I_raw.csv", ["ID"]),
    (f"{RAW_FS_DIR}/CARDIFF_PE_II_FastSurfer_V2_data.csv",
     f"{DST_DIR}/ALSPAC_30_MRI_II_raw.csv", ["ID"]),
]

# Files with no subject IDs - copy as-is.
PASSTHROUGH = [
    (f"{ALSPAC_REVIEW}/batches_7_50.csv",
     f"{DST_DIR}/batches.csv"),
    (f"{ALSPAC_REVIEW}/escaneres_shortagerange.csv",
     f"{DST_DIR}/scanners.csv"),
]


def canon_id(s) -> str:
    """Strip optional sub- / _brain / .nii suffixes; uppercase."""
    s = str(s).strip()
    s = s.replace("sub-", "").replace("SUB-", "")
    s = s.replace("_brain", "")
    s = re.sub(r"\.nii(\.gz)?$", "", s, flags=re.I)
    return s.upper()


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MAPPING_DST), exist_ok=True)

    # 1. Collect every unique canonical ID across all source files.
    all_ids = set()
    for src, _dst, id_cols in SOURCES:
        df = pd.read_csv(src, low_memory=False)
        for col in id_cols:
            if col in df.columns:
                ids = df[col].dropna().map(canon_id)
                all_ids.update(ids)
            else:
                print(f"WARN: column {col!r} missing from {src}")

    # 1b. Idempotent: load existing mapping if present, extend with any
    #     newly-seen IDs without remapping the previously-anonymised ones.
    if os.path.exists(MAPPING_DST):
        prior = pd.read_csv(MAPPING_DST)
        mapping = dict(zip(prior["real_ID"], prior["anon_ID"]))
        next_idx = 1 + max(
            (int(a.split("_")[1]) for a in mapping.values()),
            default=0,
        )
        new_ids = sorted(all_ids - set(mapping))
        # width: consistent with the existing scheme (>= 5 digits anyway)
        width = max(5, len(str(next_idx + len(new_ids) - 1)))
        for real in new_ids:
            mapping[real] = f"SUBJ_{next_idx:0{width}d}"
            next_idx += 1
        print(f"Reusing existing mapping ({len(prior)} kept) "
              f"+ {len(new_ids)} new IDs.")
    else:
        sorted_ids = sorted(all_ids)
        width = max(5, len(str(len(sorted_ids))))
        mapping = {real: f"SUBJ_{i+1:0{width}d}" for i, real in enumerate(sorted_ids)}
        print(f"Built fresh mapping for {len(mapping)} unique IDs.")

    # 2. Save the (possibly extended) mapping (gitignored).
    pd.DataFrame({
        "real_ID": list(mapping.keys()),
        "anon_ID": list(mapping.values()),
    }).to_csv(MAPPING_DST, index=False)
    print(f"Mapping saved to {MAPPING_DST}  ({len(mapping)} entries total).")

    # 3. Apply mapping (NaN-safe) and write the obfuscated copy.
    def obf(x):
        if pd.isna(x):
            return x
        return mapping.get(canon_id(x), x)

    for src, dst, id_cols in SOURCES:
        df = pd.read_csv(src, low_memory=False)
        for col in id_cols:
            if col in df.columns:
                df[col] = df[col].apply(obf)
        df.to_csv(dst, index=False)
        print(f"Wrote {dst}  (rows={len(df)})")

    # 4. Files with no IDs - straight copy.
    for src, dst in PASSTHROUGH:
        df = pd.read_csv(src)
        df.to_csv(dst, index=False)
        print(f"Wrote {dst}  (rows={len(df)}, no IDs)")


if __name__ == "__main__":
    main()
