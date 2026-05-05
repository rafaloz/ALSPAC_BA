# =============================================================================
# 08_LMM_sensitivity.R
# -----------------------------------------------------------------------------
# Sensitivity analyses for the LMM:
#   (a) Euler:      keep scans with euler_standard above the 5th percentile.
#   (b) Depression: add depression_Comp (FJCI1001 + FJCI1002) as ordinal
#                   fixed effect with PE interaction; drop NAs; compare AIC.
#
# Inputs (relative to repo root):
#   data/lmm_input_long.csv
#   data/lmm_input_long_pairs.csv
#
# Outputs:
#   data/lmm_results_euler_sensitivity.csv
#   data/lmm_results_depression_sensitivity.csv
#
# Run from the repo root:
#       Rscript src/08_LMM_sensitivity.R
# =============================================================================

suppressPackageStartupMessages({
  library(nlme)
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(effectsize)
  library(lmtest)
  library(car)
})

DATA_DIR <- "data"

LONG_CSV  <- file.path(DATA_DIR, "lmm_input_long.csv")
PAIRS_CSV <- file.path(DATA_DIR, "lmm_input_long_pairs.csv")

OUT_EULER <- file.path(DATA_DIR, "lmm_results_euler_sensitivity.csv")
OUT_DEP   <- file.path(DATA_DIR, "lmm_results_depression_sensitivity.csv")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
prep_factors <- function(d) {
  d$pliks18TH  <- factor(d$pliks18TH,  levels = c("0", "1", "2", "3"))
  d$Time       <- factor(d$Time,       levels = c("0", "1"))
  d$TimeNum    <- as.numeric(as.character(d$Time))
  d$trajectory <- factor(d$trajectory, levels = c("-1", "0", "1", "2", "3"))
  d
}

extract_contrast <- function(contr, eff = NULL, family, traj_def) {
  s <- as.data.frame(summary(contr, infer = TRUE))
  out <- data.frame(
    family    = family,
    traj_def  = traj_def,
    contrast  = s$contrast,
    estimate  = s$estimate,
    SE        = s$SE,
    df        = if ("df" %in% names(s)) s$df else NA,
    p.value   = s$p.value,
    lower.CL  = if ("lower.CL" %in% names(s)) s$lower.CL else NA,
    upper.CL  = if ("upper.CL" %in% names(s)) s$upper.CL else NA,
    stringsAsFactors = FALSE
  )
  if (!is.null(eff)) {
    es <- as.data.frame(summary(eff, infer = TRUE))
    out$cohen_d <- es$effect.size
  } else {
    out$cohen_d <- NA
  }
  out
}

run_main_contrasts <- function(fit_simple, fit_full, traj_label) {
  # fit_simple : varIdent only (for sigma(fit) at the 20-y baseline contrast,
  #              matching the published main analysis)
  # fit_full   : varComb(varIdent, varExp) (for emmeans + everything else)
  emm0 <- emmeans(fit_full, ~ pliks18TH, at = list(Time = "0"))
  emm1 <- emmeans(fit_full, ~ pliks18TH, at = list(Time = "1"))

  h1 <- list("Control vs (1+2+3)" = c(-1, 1/3, 1/3, 1/3))
  c0 <- contrast(emm0, method = h1, adjust = "none")
  c1 <- contrast(emm1, method = h1, adjust = "none")
  e0 <- eff_size(c0, sigma = sigma(fit_simple), edf = Inf,
                 method = "identity", type = "d")
  e1 <- eff_size(c1, sigma = sigma(fit_full),   edf = Inf,
                 method = "identity", type = "d")

  t0 <- contrast(emm0, "poly", adjust = "none")
  t1 <- contrast(emm1, "poly", adjust = "none")

  # Keep only the LINEAR component of the polynomial test (matches paper FDR
  # family); quadratic + cubic are auxiliary and stay separate.
  t0_lin <- as.data.frame(summary(t0))
  t0_lin <- t0[grep("linear", t0_lin$contrast), ]
  t1_lin <- as.data.frame(summary(t1))
  t1_lin <- t1[grep("linear", t1_lin$contrast), ]

  emm_long <- emmeans(fit_full, ~ Time | pliks18TH)
  delta    <- rbind(contrast(emm_long, "revpairwise"))
  div      <- contrast(delta, list("D CTRL - D PE" = c(-1, 1/3, 1/3, 1/3)),
                       adjust = "none")
  d_div    <- summary(div)$estimate / sigma(fit_simple)

  out <- rbind(
    extract_contrast(c0, e0, "H1_cross_sectional",
                     paste0(traj_label, "_T0")),
    extract_contrast(c1, e1, "H1_cross_sectional",
                     paste0(traj_label, "_T1")),
    extract_contrast(t0_lin, NULL, "H2_severity_trend_linear",
                     paste0(traj_label, "_T0")),
    extract_contrast(t1_lin, NULL, "H2_severity_trend_linear",
                     paste0(traj_label, "_T1")),
    extract_contrast(div, NULL,
                     paste0("H3_longitudinal_", traj_label),
                     traj_label)
  )
  # Attach H3 divergence d (manual; eff_size on a 1-row contrast returns
  # pairwise comparisons, not a per-row d).
  out[out$family == paste0("H3_longitudinal_", traj_label) &
      out$contrast == "D CTRL - D PE", "cohen_d"] <- d_div
  out
}

bh_within_family <- function(df) {
  df$q.value <- ave(df$p.value, df$family,
                    FUN = function(p) p.adjust(p, method = "BH"))
  df
}

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
all_data_together <- read.csv(LONG_CSV,  stringsAsFactors = FALSE)
df_long           <- read.csv(PAIRS_CSV, stringsAsFactors = FALSE)
all_data_together <- prep_factors(all_data_together)
df_long           <- prep_factors(df_long)

# =============================================================================
# (a) Euler sensitivity - drop the worst 5% of scans
# =============================================================================
cat("\n===== Euler sensitivity =====\n")

q5_all  <- quantile(all_data_together$euler_standard, 0.05, na.rm = TRUE)
ad_eul  <- subset(all_data_together, euler_standard >= q5_all)

q5_long <- quantile(df_long$euler_standard, 0.05, na.rm = TRUE)
df_eul  <- subset(df_long, euler_standard >= q5_long)

cat("Removed N=", nrow(all_data_together) - nrow(ad_eul),
    " rows from cross-sectional set; ",
    nrow(df_long) - nrow(df_eul),
    " from paired set.\n", sep = "")

# Euler sensitivity uses the MAIN model (no depression covariate); only the
# data subset changes (worst 5% Euler scans dropped).
fit_eul_simple <- lme(
  BrainPAD_YJ_c ~ Time * pliks18TH + sexo + euler_n,
  random      = ~ Time | ID,
  weights     = varIdent(form = ~ 1 | pliks18TH),
  data        = ad_eul,
  method      = "REML",
  na.action   = na.omit
)
fit_eul <- update(fit_eul_simple,
                  weights = varComb(varIdent(form = ~ 1 | pliks18TH),
                                    varExp(form  = ~ abs(fitted(.)))))

res_eul1 <- run_main_contrasts(fit_eul_simple, fit_eul, "LPEs1_euler")

fit_traj_eul <- lmer(
  BrainPAD_YJ_c ~ TimeNum * trajectory + sexo + euler_n + (1 | ID),
  data = df_eul, REML = FALSE
)
emm_traj <- emmeans(fit_traj_eul, ~ TimeNum | trajectory,
                    at = list(TimeNum = c(0, 1)))
delta2   <- rbind(contrast(emm_traj, "revpairwise"))
div2     <- contrast(delta2,
                     list("D CTRL - D PE" = c(-1, 1/3, 1/3, 1/3)),
                     adjust = "none")
res_eul2 <- extract_contrast(div2, NULL,
                             "H3_longitudinal_LPEs2_euler",
                             "LPEs2_euler")
res_eul2[res_eul2$contrast == "D CTRL - D PE", "cohen_d"] <-
  summary(div2)$estimate / sigma(fit_traj_eul)

res_eul <- bh_within_family(rbind(res_eul1, res_eul2))
res_eul$AIC_full <- AIC(fit_eul)
write.csv(res_eul, OUT_EULER, row.names = FALSE)
cat("Wrote ", OUT_EULER, "\n", sep = "")

# =============================================================================
# (b) Depression sensitivity - add depression_Comp + interaction
# =============================================================================
cat("\n===== Depression sensitivity =====\n")

ad_dep  <- subset(all_data_together,
                  !is.na(FJCI1001) & !is.na(FJCI1002) & !is.na(depression_Comp))
df_dep  <- subset(df_long, !is.na(depression_Comp))

# Source line 981 treats depression_Comp as numeric (0/1/2 = sum of two
# binary indicator variables). Keep as numeric here for parity.
ad_dep$depression_Comp <- as.numeric(ad_dep$depression_Comp)
df_dep$depression_Comp <- as.numeric(df_dep$depression_Comp)

# NOTE: paper Table S/Discussion reports ΔAIC = +5.9 between
# fit_with_dep and fit_no_dep. That number reproduces only with
# method="REML" + varComb weights (the published number is technically
# incorrect for comparing models that differ in fixed effects, but it is
# what the paper printed). Using method="ML" gives ΔAIC ≈ -2.3.
fit_no_dep <- lme(
  BrainPAD_YJ_c ~ Time * pliks18TH + sexo + euler_n,
  random      = ~ Time | ID,
  weights     = varComb(varIdent(form = ~ 1 | pliks18TH),
                        varExp(form  = ~ abs(fitted(.)))),
  data        = ad_dep,
  method      = "REML",
  na.action   = na.omit
)
fit_with_dep_simple <- lme(
  BrainPAD_YJ_c ~ Time * pliks18TH + depression_Comp * pliks18TH +
                  sexo + euler_n,
  random      = ~ Time | ID,
  weights     = varIdent(form = ~ 1 | pliks18TH),
  data        = ad_dep,
  method      = "REML",
  na.action   = na.omit
)
fit_with_dep <- update(fit_with_dep_simple,
                       weights = varComb(varIdent(form = ~ 1 | pliks18TH),
                                         varExp(form  = ~ abs(fitted(.)))))
cat("AIC no-dep =", AIC(fit_no_dep),
    "  AIC with-dep =", AIC(fit_with_dep),
    "  Delta AIC (with - no) =", AIC(fit_with_dep) - AIC(fit_no_dep), "\n")

res_dep1 <- run_main_contrasts(fit_with_dep_simple, fit_with_dep, "LPEs1_dep")

fit_traj_dep <- lmer(
  BrainPAD_YJ_c ~ TimeNum * trajectory + depression_Comp * trajectory +
                  sexo + euler_n + (1 | ID),
  data = df_dep, REML = FALSE
)
emm_traj <- emmeans(fit_traj_dep, ~ TimeNum | trajectory,
                    at = list(TimeNum = c(0, 1)))
delta2   <- rbind(contrast(emm_traj, "revpairwise"))
div2     <- contrast(delta2,
                     list("D CTRL - D PE" = c(-1, 1/3, 1/3, 1/3)),
                     adjust = "none")
res_dep2 <- extract_contrast(div2, NULL,
                             "H3_longitudinal_LPEs2_dep", "LPEs2_dep")
res_dep2[res_dep2$contrast == "D CTRL - D PE", "cohen_d"] <-
  summary(div2)$estimate / sigma(fit_traj_dep)

# Depression MAIN-effect Cohen's d (paper Table S6 / Discussion):
# beta_depression / sigma(fit_with_dep) — the varComb fit, NOT varIdent-only,
# matching the published d=0.23 [0.03, 0.43], p=.027.
co_dep <- summary(fit_with_dep)$tTable
ci_dep <- intervals(fit_with_dep, which = "fixed")$fixed
sg_dep <- sigma(fit_with_dep)
dep_row <- data.frame(
  family    = "depression_main_effect",
  traj_def  = "LPEs1_dep",
  contrast  = "depression_Comp main",
  estimate  = co_dep["depression_Comp", "Value"],
  SE        = co_dep["depression_Comp", "Std.Error"],
  df        = co_dep["depression_Comp", "DF"],
  p.value   = co_dep["depression_Comp", "p-value"],
  lower.CL  = ci_dep["depression_Comp", "lower"],
  upper.CL  = ci_dep["depression_Comp", "upper"],
  cohen_d   = co_dep["depression_Comp", "Value"] / sg_dep,
  q.value   = co_dep["depression_Comp", "p-value"],
  stringsAsFactors = FALSE
)
cat(sprintf("Depression main effect d=%.3f [%.3f, %.3f], p=%.4f (paper: 0.23 [0.03, 0.43], p=.027)\n",
            dep_row$cohen_d,
            ci_dep["depression_Comp", "lower"] / sg_dep,
            ci_dep["depression_Comp", "upper"] / sg_dep,
            dep_row$p.value))

res_dep <- bh_within_family(rbind(res_dep1, res_dep2))
# Pad columns so depression-main row aligns with res_dep schema
for (col in setdiff(names(res_dep), names(dep_row))) dep_row[[col]] <- NA
for (col in setdiff(names(dep_row), names(res_dep))) res_dep[[col]] <- NA
res_dep <- rbind(res_dep, dep_row[, names(res_dep)])
res_dep$AIC_no_dep   <- AIC(fit_no_dep)
res_dep$AIC_with_dep <- AIC(fit_with_dep)
res_dep$delta_AIC    <- AIC(fit_no_dep) - AIC(fit_with_dep)
write.csv(res_dep, OUT_DEP, row.names = FALSE)
cat("Wrote ", OUT_DEP, "\n", sep = "")

cat("\nDone.\n")
