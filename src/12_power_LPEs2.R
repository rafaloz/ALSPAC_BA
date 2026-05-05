###############################################################################
##  12_power_LPEs2.R — Monte-Carlo power analysis for the LPEs-2 LMM
##                     (longitudinal subset under trajectory)
##
##  Reproduces supplement Table S18 (LPEs-2 family).
##
##  Inputs (relative to repo root):
##      data/lmm_input_long_pairs.csv     (df_long; produced by
##                                         src/07_export_LMM_input.py)
##
##  Outputs:
##      data/power_LPEs2.csv              (power per hypothesis, alpha=0.05)
##
##  Run from the repo root:
##      Rscript src/12_power_LPEs2.R
###############################################################################

library(nlme)
library(lme4)
library(emmeans)
library(future.apply)
library(MASS)

DATA_DIR <- "data"
OUT_CSV  <- file.path(DATA_DIR, "power_LPEs2.csv")

###############################################################################
##  0)  Prepare data
###############################################################################
all_data_together <- read.csv(file.path(DATA_DIR, "lmm_input_long_pairs.csv"))

all_data_together$pliks18TH  <- factor(all_data_together$pliks18TH, levels = c("0","1","2","3"))
all_data_together$sexo       <- as.numeric(all_data_together$sexo)
all_data_together$euler_n    <- scale(all_data_together$euler_n, center = TRUE, scale = TRUE)
all_data_together$trajectory <- factor(all_data_together$trajectory, levels = c("0","1","2","3"))
all_data_together$Time       <- factor(all_data_together$Time, levels = c(0, 1))
all_data_together$TimeNum    <- as.numeric(as.character(all_data_together$Time))

###############################################################################
##  0)  USER-TUNE — design and "true" parameter values
###############################################################################
N20  <- c(124, 41, 45, 35)
N30  <- c(210, 29, 26, 14)
NREP <- 113

beta_time   <-  0.20
beta_pliks  <- c(0, .3, .6, .9)
beta_sex    <-  0.10
beta_euler  <- -0.05

sd_intercept <- 1.00
sd_slope     <- 0.30
rho_IS       <- 0.10

sigma_eps    <- c(1.00, 1.10, 1.15, 1.20)

NSIM  <- 1000
ALPHA <- 0.05
NCORE <- parallel::detectCores() - 1

###############################################################################
##  1)  helper: simulate one synthetic data set
###############################################################################
simulate_data <- function(seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  rows <- list();  id <- 0L

  pl20 <- sample(0:3, NREP, replace = TRUE, prob = N20 / sum(N20))
  for (pl in pl20) {
    rows[[length(rows) + 1]] <- data.frame(ID = id, Time = "0", pliks18TH = pl)
    rows[[length(rows) + 1]] <- data.frame(ID = id, Time = "1", pliks18TH = pl)
    id <- id + 1L
  }
  for (pl in 0:3) {
    need <- N20[pl + 1] - sum(pl20 == pl)
    if (need > 0)
      rows[[length(rows) + 1]] <- data.frame(
        ID = id + (0:(need - 1)), Time = "0", pliks18TH = pl)
    id <- id + need
  }
  for (pl in 0:3) {
    need <- N30[pl + 1]
    rows[[length(rows) + 1]] <- data.frame(
      ID = id + (0:(need - 1)), Time = "1", pliks18TH = pl)
    id <- id + need
  }

  df <- do.call(rbind, rows)
  df$ID      <- factor(df$ID)
  df$Time    <- factor(df$Time, levels = c("0", "1"))
  df$TimeNum <- as.numeric(as.character(df$Time))

  pick <- sample.int(nrow(all_data_together), nrow(df), replace = TRUE)
  df$sexo    <- all_data_together$sexo   [pick]
  df$euler_n <- all_data_together$euler_n[pick]

  Sigma <- matrix(c(sd_intercept^2,
                    rho_IS * sd_intercept * sd_slope,
                    rho_IS * sd_intercept * sd_slope,
                    sd_slope^2), 2)
  ranef <- MASS::mvrnorm(nlevels(df$ID), mu = c(0, 0), Sigma)
  colnames(ranef) <- c("u0", "u1")
  df <- merge(df, data.frame(ID = levels(df$ID), ranef), by = "ID")

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
##  2)  helper: analyse one data set -> five p-values (lme4)
###############################################################################
analyse_once <- function(df) {
  fit <- lmer(
    BrainPAD_YJ_c ~ Time * pliks18TH + sexo + euler_n + (1 | ID),
    data    = df,
    REML    = FALSE
  )

  emm0 <- emmeans(fit, ~ pliks18TH, at = list(Time = "0"))
  emm1 <- emmeans(fit, ~ pliks18TH, at = list(Time = "1"))

  cvec <- c(-1, 1/3, 1/3, 1/3); names(cvec) <- levels(df$pliks18TH)
  H1_base   <- contrast(emm0, list(ctrl_vs_pooled = cvec), adjust = "none")
  H1_follow <- contrast(emm1, list(ctrl_vs_pooled = cvec), adjust = "none")

  poly <- c(-3, -1, 1, 3) / 10; names(poly) <- levels(df$pliks18TH)
  H2_base   <- contrast(emm0, list(trend = poly), adjust = "none")
  H2_follow <- contrast(emm1, list(trend = poly), adjust = "none")

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

cat(sprintf("\nPower (alpha = %.2f, %s simulations) — LPEs-2\n",
            ALPHA, format(NSIM, big.mark = ",")))
print(round(power, 3))

write.csv(data.frame(hypothesis = names(power), power = round(power, 3)),
          OUT_CSV, row.names = FALSE)
cat("Wrote ", OUT_CSV, "\n", sep = "")
