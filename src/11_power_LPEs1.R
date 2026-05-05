###############################################################################
##  11_power_LPEs1.R — Monte-Carlo power analysis for the LPEs-1 LMM
##                     (cross-sectional + longitudinal under pliks18TH)
##
##  Reproduces supplement Table S18 (LPEs-1 family).
##
##  Inputs (relative to repo root):
##      data/lmm_input_long.csv          (all_data_together; produced by
##                                        src/07_export_LMM_input.py)
##
##  Outputs:
##      data/power_LPEs1.csv             (power per hypothesis, alpha=0.05)
##
##  Run from the repo root:
##      Rscript src/11_power_LPEs1.R
###############################################################################

library(nlme)            # lme()
library(emmeans)         # contrasts
library(future.apply)    # parallel loops
library(MASS)            # mvrnorm

DATA_DIR <- "data"
OUT_CSV  <- file.path(DATA_DIR, "power_LPEs1.csv")

###############################################################################
##  0)  Prepare data
###############################################################################
all_data_together <- read.csv(file.path(DATA_DIR, "lmm_input_long.csv"))

all_data_together$pliks18TH  <- factor(all_data_together$pliks18TH, levels = c("0","1","2","3"))
all_data_together$sexo       <- as.numeric(all_data_together$sexo)
all_data_together$euler_n    <- scale(all_data_together$euler_n, center = TRUE, scale = TRUE)
all_data_together$trajectory <- factor(all_data_together$trajectory, levels = c("0","1","2","3"))
all_data_together$Time       <- factor(all_data_together$Time, levels = c(0, 1))
all_data_together$TimeNum    <- as.numeric(as.character(all_data_together$Time))

###############################################################################
##  0)  USER-TUNE — design and "true" parameter values
###############################################################################
## 0-a sample-size skeleton (observed in this study)
N20  <- c(124, 41, 45, 35)            # pliks 0..3 at baseline (20y)
N30  <- c(210, 29, 26, 14)            # pliks 0..3 at follow-up (30y)
NREP <- 113                           # subjects measured at both waves

## 0-b fixed-effect sizes you want power for (SD units)
beta_time   <-  0.20                       # overall ageing shift
beta_pliks  <- c(0, .3, .6, .9)            # means of pliks 0..3 at 20y
beta_sex    <-  0.10                       # male - female
beta_euler  <- -0.05                       # per SD of Euler

## 0-c random-effect covariance (intercept & slope)
sd_intercept <- 1.00
sd_slope     <- 0.30
rho_IS       <- 0.10                       # corr(intercept, slope)

## 0-d residual SD per pliks level (implements varIdent)
sigma_eps    <- c(1.00, 1.10, 1.15, 1.20)

## 0-e simulation control
NSIM  <- 1000
ALPHA <- 0.05
NCORE <- parallel::detectCores() - 1       # leave one core free

###############################################################################
##  1)  helper: simulate one synthetic data set
###############################################################################
simulate_data <- function(seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  rows <- list();  id <- 0L

  ## 1-A repeated participants (2 rows each)
  pl20 <- sample(0:3, NREP, replace = TRUE, prob = N20 / sum(N20))
  for (pl in pl20) {
    rows[[length(rows) + 1]] <- data.frame(ID = id, Time = "0", pliks18TH = pl)
    rows[[length(rows) + 1]] <- data.frame(ID = id, Time = "1", pliks18TH = pl)
    id <- id + 1L
  }

  ## 1-B 20y-only cohort
  for (pl in 0:3) {
    need <- N20[pl + 1] - sum(pl20 == pl)
    if (need > 0)
      rows[[length(rows) + 1]] <- data.frame(
        ID = id + (0:(need - 1)), Time = "0", pliks18TH = pl)
    id <- id + need
  }

  ## 1-C 30y-only cohort
  for (pl in 0:3) {
    need <- N30[pl + 1]
    rows[[length(rows) + 1]] <- data.frame(
      ID = id + (0:(need - 1)), Time = "1", pliks18TH = pl)
    id <- id + need
  }

  ## 1-D bind & add covariates (sampled with replacement from observed data)
  df <- do.call(rbind, rows)
  df$ID   <- factor(df$ID)
  df$Time <- factor(df$Time, levels = c("0", "1"))

  pick <- sample.int(nrow(all_data_together), nrow(df), replace = TRUE)
  df$sexo    <- all_data_together$sexo   [pick]
  df$euler_n <- all_data_together$euler_n[pick]

  ## 1-E random intercept & slope ~ N(0, Sigma)
  Sigma <- matrix(c(sd_intercept^2,
                    rho_IS * sd_intercept * sd_slope,
                    rho_IS * sd_intercept * sd_slope,
                    sd_slope^2), 2)
  ranef <- MASS::mvrnorm(nlevels(df$ID), mu = c(0, 0), Sigma)
  colnames(ranef) <- c("u0", "u1")
  df <- merge(df, data.frame(ID = levels(df$ID), ranef), by = "ID")

  ## 1-F true mean mu
  mu <- beta_pliks[df$pliks18TH + 1] +
        beta_time  * as.numeric(as.character(df$Time)) +
        beta_sex   * df$sexo +
        beta_euler * df$euler_n

  df$BrainPAD_YJ_c <- mu +
    df$u0 +
    df$u1 * as.numeric(as.character(df$Time)) +
    rnorm(nrow(df), 0, sigma_eps[df$pliks18TH + 1])

  df$pliks18TH <- factor(df$pliks18TH, levels = 0:3)
  return(df)
}

###############################################################################
##  2)  helper: analyse one data set -> five p-values
###############################################################################
analyse_once <- function(df) {
  fit <- lme(
    BrainPAD_YJ_c ~ Time * pliks18TH + sexo + euler_n,
    random   = ~ Time | ID,
    weights  = varIdent(form = ~ 1 | pliks18TH),
    data     = df,
    method   = "REML",
    control  = lmeControl(opt = "optim", msMaxIter = 50, tolerance = 1e-6)
  )

  emm0 <- emmeans(fit, ~ pliks18TH, at = list(Time = "0"))
  emm1 <- emmeans(fit, ~ pliks18TH, at = list(Time = "1"))

  ## H1: control vs pooled 1-3
  cvec <- c(-1, 1/3, 1/3, 1/3); names(cvec) <- levels(df$pliks18TH)
  H1_base   <- contrast(emm0, list(ctrl_vs_pooled = cvec), adjust = "none")
  H1_follow <- contrast(emm1, list(ctrl_vs_pooled = cvec), adjust = "none")

  ## H2: linear 0->3 trend
  poly <- c(-3, -1, 1, 3) / 10; names(poly) <- levels(df$pliks18TH)
  H2_base   <- contrast(emm0, list(trend = poly), adjust = "none")
  H2_follow <- contrast(emm1, list(trend = poly), adjust = "none")

  ## H3: ΔCTRL - Δpooled
  emm_time <- emmeans(fit, ~ Time | pliks18TH)
  delta <- contrast(emm_time, "revpairwise") |> rbind()
  H3 <- contrast(delta, list(divergence = c(-1, 1/3, 1/3, 1/3)),
                 adjust = "none")

  c(H1_base   = summary(H1_base)$p.value,
    H1_follow = summary(H1_follow)$p.value,
    H2_base   = summary(H2_base)$p.value,
    H2_follow = summary(H2_follow)$p.value,
    H3        = summary(H3)$p.value)
}

###############################################################################
##  3)  MONTE-CARLO LOOP
###############################################################################
plan(multisession, workers = NCORE)
set.seed(1)
pvals <- future_replicate(NSIM, {
  df <- simulate_data()
  analyse_once(df)
}, future.seed = TRUE, simplify = "matrix")
power <- rowMeans(pvals < ALPHA)

cat(sprintf("\nPower (alpha = %.2f, %s simulations) — LPEs-1\n",
            ALPHA, format(NSIM, big.mark = ",")))
print(round(power, 3))

write.csv(data.frame(hypothesis = names(power), power = round(power, 3)),
          OUT_CSV, row.names = FALSE)
cat("Wrote ", OUT_CSV, "\n", sep = "")
