# =============================================================================
# 07_LMM_main.R
# -----------------------------------------------------------------------------
# Linear mixed-model analysis for ALSPAC BrainPAD ~ PE status.
#
# Inputs (relative to repo root):
#   data/lmm_input_long.csv         (all_data_together; one row per visit)
#   data/lmm_input_long_pairs.csv   (df_long: subjects with both visits)
#
# Outputs:
#   data/lmm_results_main.csv       contrasts (Δ, SE, p, q, Cohen's d, 95% CI)
#   data/lmm_diagnostics_main.csv   Shapiro / Breusch-Pagan / AIC / BIC
#   figures/residuals_vs_fitted.svg
#   figures/qqplot_residuals.svg
#   figures/scale_location_plot.svg
#   figures/residuals_vs_fitted_2.svg
#   figures/qqplot_residuals_2.svg
#   figures/scale_location_plot_2.svg
#
# Run from the repo root:
#       Rscript src/07_LMM_main.R
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

# -----------------------------------------------------------------------------
# 0. Path setup
# -----------------------------------------------------------------------------
DATA_DIR <- "data"
FIG_DIR  <- "figures"
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

LONG_CSV       <- file.path(DATA_DIR, "lmm_input_long.csv")
PAIRS_CSV      <- file.path(DATA_DIR, "lmm_input_long_pairs.csv")
OUT_RESULTS    <- file.path(DATA_DIR, "lmm_results_main.csv")
OUT_DIAG       <- file.path(DATA_DIR, "lmm_diagnostics_main.csv")

# -----------------------------------------------------------------------------
# 1. Load
# -----------------------------------------------------------------------------
all_data_together <- read.csv(LONG_CSV, stringsAsFactors = FALSE)
df_long           <- read.csv(PAIRS_CSV, stringsAsFactors = FALSE)

# Factor coding mirroring the source (lines 967-972).
all_data_together$pliks18TH  <- factor(all_data_together$pliks18TH,
                                       levels = c("0", "1", "2", "3"))
all_data_together$Time       <- factor(all_data_together$Time, levels = c("0", "1"))
all_data_together$TimeNum    <- as.numeric(as.character(all_data_together$Time))
all_data_together$trajectory <- factor(all_data_together$trajectory,
                                       levels = c("-1", "0", "1", "2", "3"))
all_data_together$Edad_c     <- scale(all_data_together$Edad,
                                      center = TRUE, scale = FALSE)

df_long$Time       <- factor(df_long$Time, levels = c("0", "1"))
df_long$TimeNum    <- as.numeric(as.character(df_long$Time))
df_long$trajectory <- factor(df_long$trajectory, levels = c("0", "1", "2", "3"))

cat("=== Time x pliks18TH ===\n")
print(with(all_data_together, table(Time, pliks18TH)))

# -----------------------------------------------------------------------------
# 2. Cross-sectional + longitudinal LMM (LPEs-1 trajectory = pliks18TH)
#    Source lines 981-982.
# -----------------------------------------------------------------------------
fit <- lme(
  BrainPAD_YJ_c ~ Time * pliks18TH + sexo + euler_n,
  random      = ~ Time | ID,
  weights     = varIdent(form = ~ 1 | pliks18TH),
  data        = all_data_together,
  method      = "REML",
  na.action   = na.omit
)

fit1 <- update(
  fit,
  weights = varComb(varIdent(form = ~ 1 | pliks18TH),
                    varExp(form  = ~ abs(fitted(.))))
)

cat("AIC(fit1) =", AIC(fit1), "  BIC(fit1) =", BIC(fit1), "\n")
print(summary(fit1))

# -----------------------------------------------------------------------------
# 3. Diagnostics (LPEs-1)
# -----------------------------------------------------------------------------
diag_rows <- list()

shap1 <- shapiro.test(residuals(fit1, type = "pearson"))
lm_like <- lm(residuals(fit1, type = "pearson") ~ fitted(fit1))
bp1   <- bptest(lm_like)
diag_rows[["LPEs1"]] <- data.frame(
  model           = "fit1_LPEs1",
  AIC             = AIC(fit1),
  BIC             = BIC(fit1),
  shapiro_W       = shap1$statistic,
  shapiro_p       = shap1$p.value,
  bp_LM           = bp1$statistic,
  bp_p            = bp1$p.value,
  stringsAsFactors = FALSE
)

svg(file.path(FIG_DIR, "residuals_vs_fitted.svg"), width = 7, height = 7)
plot(fitted(fit1), residuals(fit1, type = "pearson"),
     xlab = "Fitted values", ylab = "Residuals",
     main = "Residuals vs Fitted Values")
abline(h = 0, col = "red", lty = 2)
dev.off()

svg(file.path(FIG_DIR, "qqplot_residuals.svg"), width = 7, height = 7)
qqnorm(residuals(fit1), main = "Normal Q-Q Plot of Residuals",
       ylab = "Residuals")
qqline(residuals(fit1), col = "red", lty = 2)
dev.off()

svg(file.path(FIG_DIR, "scale_location_plot.svg"), width = 7, height = 7)
resid_std <- residuals(fit1, type = "pearson")
plot(fitted(fit1), sqrt(abs(resid_std)),
     xlab = "Fitted values", ylab = "sqrt(|Std residuals|)",
     main = "Scale-Location Plot")
dev.off()

# -----------------------------------------------------------------------------
# 4. EMMs and contrasts (LPEs-1, fit1)
# -----------------------------------------------------------------------------
emm_time0 <- emmeans(fit1, ~ pliks18TH, at = list(Time = "0"))
emm_time1 <- emmeans(fit1, ~ pliks18TH, at = list(Time = "1"))

# H1: Control vs (1+2+3) at each visit
h1_contrast <- list("Control vs (1+2+3)" = c(-1, 1/3, 1/3, 1/3))
ctrl_t0 <- contrast(emm_time0, method = h1_contrast, adjust = "none")
ctrl_t1 <- contrast(emm_time1, method = h1_contrast, adjust = "none")

# NOTE on Cohen's d denominators ----------------------------------------------
# The published Table 2 uses sigma(fit) for the 20-year baseline contrast and
# sigma(fit1) for the 30-year follow-up contrast (source script
# `statisticsReview_7_50_Clean.py`, lines 1123 and 1139 respectively).
# `fit` carries only varIdent weights; `fit1` adds varExp(~|fitted|), so the
# two residual scales are NOT directly comparable.
#
# Consequence: the d values printed for H1 baseline (20y) and H1 follow-up
# (30y) are each individually correct under their own sigma reference, but
# the cross-row comparison ("d shrinks from 0.70 to 0.22") conflates the
# change in sigma reference with the change in effect.
#
# This block reproduces the paper verbatim. To recompute with a single sigma,
# set both eff_size() calls to use sigma(fit) or sigma(fit1) consistently.
# -----------------------------------------------------------------------------
eff_t0 <- eff_size(ctrl_t0, sigma = sigma(fit),  edf = Inf,
                   method = "identity", type = "d")
eff_t1 <- eff_size(ctrl_t1, sigma = sigma(fit1), edf = Inf,
                   method = "identity", type = "d")

# H2: ordinal poly trend at each visit
trend_t0 <- contrast(emm_time0, method = "poly", adjust = "none")
trend_t1 <- contrast(emm_time1, method = "poly", adjust = "none")
sig <- summary(fit1)$sigma
edf <- summary(trend_t0)$df[1]
d_trend_t0 <- eff_size(trend_t0, sigma = sig, edf = edf,
                       type = "d", method = "identity")
d_trend_t1 <- eff_size(trend_t1, sigma = sig, edf = edf,
                       type = "d", method = "identity")

# H3: longitudinal divergence
emm_long <- emmeans(fit1, ~ Time | pliks18TH)
delta    <- rbind(contrast(emm_long, "revpairwise"))
div      <- contrast(delta, list("D CTRL - D PE" = c(-1, 1/3, 1/3, 1/3)),
                     adjust = "none")
d_delta  <- eff_size(delta, sigma = summary(fit1)$sigma,
                     edf = 108, type = "d")
# Cohen's d for the divergence contrast itself, using sigma(fit)
# (source line 1220).
d_div_LPEs1 <- summary(div)$estimate / sigma(fit)
cat(sprintf("Cohen's d for H3 LPEs-1 divergence: %.3f (sigma(fit)=%.3f)\n",
            d_div_LPEs1, sigma(fit)))

# -----------------------------------------------------------------------------
# 5. LPEs-2 LMM on the paired-visits frame (df_long). Source line 1278.
# -----------------------------------------------------------------------------
fit_traj <- lmer(
  BrainPAD_YJ_c ~ TimeNum * trajectory + sexo + euler_n + (1 | ID),
  data    = df_long,
  REML    = FALSE
)
print(summary(fit_traj))

shap2 <- shapiro.test(residuals(fit_traj))
bp2   <- bptest(lm(residuals(fit_traj) ~ fitted(fit_traj)))
diag_rows[["LPEs2"]] <- data.frame(
  model           = "fit_traj_LPEs2",
  AIC             = AIC(fit_traj),
  BIC             = BIC(fit_traj),
  shapiro_W       = shap2$statistic,
  shapiro_p       = shap2$p.value,
  bp_LM           = bp2$statistic,
  bp_p            = bp2$p.value,
  stringsAsFactors = FALSE
)

svg(file.path(FIG_DIR, "residuals_vs_fitted_2.svg"), width = 7, height = 7)
plot(fitted(fit_traj), residuals(fit_traj),
     xlab = "Fitted values", ylab = "Residuals",
     main = "Residuals vs Fitted Values")
abline(h = 0, col = "red", lty = 2)
dev.off()

svg(file.path(FIG_DIR, "qqplot_residuals_2.svg"), width = 7, height = 7)
qqnorm(residuals(fit_traj), main = "Normal Q-Q Plot of Residuals",
       ylab = "Residuals")
qqline(residuals(fit_traj), col = "red", lty = 2)
dev.off()

svg(file.path(FIG_DIR, "scale_location_plot_2.svg"), width = 7, height = 7)
plot(fitted(fit_traj), sqrt(abs(residuals(fit_traj))),
     xlab = "Fitted values", ylab = "sqrt(|Std residuals|)",
     main = "Scale-Location Plot")
dev.off()

# LPEs-2 contrasts
emm_traj <- emmeans(fit_traj, ~ TimeNum | trajectory,
                    at = list(TimeNum = c(0, 1)))
delta2   <- rbind(contrast(emm_traj, "revpairwise"))
div2     <- contrast(delta2,
                     list("D CTRL - D PE" = c(-1, 1/3, 1/3, 1/3)),
                     adjust = "none")
d_delta2  <- eff_size(delta2, sigma = sigma(fit_traj), edf = Inf, type = "d")
d_div_LPEs2 <- summary(div2)$estimate / sigma(fit_traj)
cat(sprintf("Cohen's d for H3 LPEs-2 divergence: %.3f (sigma(fit_traj)=%.3f)\n",
            d_div_LPEs2, sigma(fit_traj)))

trends2  <- emtrends(fit_traj, ~ trajectory, var = "TimeNum")
H3_contr <- contrast(trends2, method = "pairwise", ref = 1, adjust = "none")
poly_traj <- contrast(trends2, "poly")

# -----------------------------------------------------------------------------
# 6. Assemble the results table
# -----------------------------------------------------------------------------
extract_contrast <- function(contr, eff = NULL, family, traj_def) {
  s <- as.data.frame(summary(contr, infer = TRUE))
  out <- data.frame(
    family    = family,
    traj_def  = traj_def,
    contrast  = s$contrast,
    estimate  = s$estimate,
    SE        = s$SE,
    df        = if ("df" %in% names(s)) s$df else NA,
    t.ratio   = if ("t.ratio" %in% names(s)) s$t.ratio else NA,
    p.value   = s$p.value,
    lower.CL  = if ("lower.CL" %in% names(s)) s$lower.CL else NA,
    upper.CL  = if ("upper.CL" %in% names(s)) s$upper.CL else NA,
    stringsAsFactors = FALSE
  )
  if (!is.null(eff)) {
    es <- as.data.frame(summary(eff, infer = TRUE))
    if (nrow(es) == nrow(out)) {
      out$cohen_d <- es$effect.size
      out$d_lower <- if ("lower.CL" %in% names(es)) es$lower.CL else NA
      out$d_upper <- if ("upper.CL" %in% names(es)) es$upper.CL else NA
    } else {
      # `eff_size` on multi-row contrasts returns pairwise comparisons
      # of effect sizes — different length from `out`. Skip in that case.
      out$cohen_d <- NA; out$d_lower <- NA; out$d_upper <- NA
    }
  } else {
    out$cohen_d <- NA; out$d_lower <- NA; out$d_upper <- NA
  }
  out
}

res <- rbind(
  extract_contrast(ctrl_t0, eff_t0,    "H1_cross_sectional", "LPEs1_T0"),
  extract_contrast(ctrl_t1, eff_t1,    "H1_cross_sectional", "LPEs1_T1"),
  extract_contrast(trend_t0, d_trend_t0, "H2_severity_trend", "LPEs1_T0"),
  extract_contrast(trend_t1, d_trend_t1, "H2_severity_trend", "LPEs1_T1"),
  extract_contrast(div,     NULL,      "H3_longitudinal",    "LPEs1"),
  extract_contrast(delta,   d_delta,   "H3_within_group_change", "LPEs1"),
  extract_contrast(div2,    NULL,      "H3_longitudinal",    "LPEs2"),
  extract_contrast(delta2,  d_delta2,  "H3_within_group_change", "LPEs2"),
  extract_contrast(H3_contr, NULL,     "H3_pairwise_vs_ctrl","LPEs2"),
  extract_contrast(poly_traj, NULL,    "H2_severity_trend_LPEs2_slope", "LPEs2_slope")
)

# -----------------------------------------------------------------------------
# 6b. Refine FDR families to match the paper's grouping:
#   * H2 paper FDRs ONLY the two linear contrasts (20y + 30y); the
#     quadratic/cubic poly components are kept separate.
#   * H3 paper FDRs within each LPE definition separately (1 test each),
#     so q == p for both rows.
# -----------------------------------------------------------------------------
res$family <- ifelse(res$family == "H2_severity_trend" &
                       grepl("linear", res$contrast),
                     "H2_severity_trend_linear",
                     res$family)
res$family <- ifelse(res$family == "H2_severity_trend" &
                       grepl("quadratic|cubic", res$contrast),
                     "H2_severity_trend_higher_order",
                     res$family)
res$family <- ifelse(res$family == "H3_longitudinal",
                     paste0("H3_longitudinal_", res$traj_def),
                     res$family)

# Attach H3 divergence Cohen's d (manually computed from sigma(fit) /
# sigma(fit_traj), since eff_size on a single-row contrast returns
# pairwise comparisons of effect sizes, not a per-row d).
res[res$family == "H3_longitudinal_LPEs1" &
    res$contrast == "D CTRL - D PE", "cohen_d"] <- d_div_LPEs1
res[res$family == "H3_longitudinal_LPEs2" &
    res$contrast == "D CTRL - D PE", "cohen_d"] <- d_div_LPEs2

# -----------------------------------------------------------------------------
# 7. Benjamini-Hochberg FDR within each hypothesis family
# -----------------------------------------------------------------------------
res$q.value <- ave(res$p.value, res$family, FUN = function(p) p.adjust(p, method = "BH"))

write.csv(res, OUT_RESULTS, row.names = FALSE)
cat("Wrote ", OUT_RESULTS, " (", nrow(res), " rows)\n", sep = "")

diag_df <- do.call(rbind, diag_rows)
write.csv(diag_df, OUT_DIAG, row.names = FALSE)
cat("Wrote ", OUT_DIAG, "\n", sep = "")

# -----------------------------------------------------------------------------
# 8. Supplementary tables S7-S11: full EMMs, pairwise contrasts, within-group
#    changes per group.
# -----------------------------------------------------------------------------
emm0_full <- as.data.frame(summary(emm_time0, infer = TRUE))
emm0_full$visit <- "20y"
emm1_full <- as.data.frame(summary(emm_time1, infer = TRUE))
emm1_full$visit <- "30y"
emm_full <- rbind(emm0_full, emm1_full)
write.csv(emm_full,
          file.path(DATA_DIR, "lmm_emmeans_full.csv"),
          row.names = FALSE)
cat("Wrote ", file.path(DATA_DIR, "lmm_emmeans_full.csv"), "\n", sep = "")

pairs0 <- as.data.frame(summary(pairs(emm_time0, adjust = "fdr"),
                                infer = TRUE))
pairs0$visit <- "20y"
pairs1 <- as.data.frame(summary(pairs(emm_time1, adjust = "fdr"),
                                infer = TRUE))
pairs1$visit <- "30y"
pairs_all <- rbind(pairs0, pairs1)
write.csv(pairs_all,
          file.path(DATA_DIR, "lmm_pairwise_full.csv"),
          row.names = FALSE)
cat("Wrote ", file.path(DATA_DIR, "lmm_pairwise_full.csv"), "\n", sep = "")

delta_full <- as.data.frame(summary(delta, infer = TRUE))
write.csv(delta_full,
          file.path(DATA_DIR, "lmm_within_group_changes.csv"),
          row.names = FALSE)
cat("Wrote ", file.path(DATA_DIR, "lmm_within_group_changes.csv"), "\n",
    sep = "")
