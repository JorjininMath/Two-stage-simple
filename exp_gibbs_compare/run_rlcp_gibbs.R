#!/usr/bin/env Rscript
# run_rlcp_gibbs.R
#
# Run RLCP (Hore & Barber 2023) on the Gibbs et al. DGPs (Setting 1 & 2),
# output local coverage AND mean width at 21 centers in [-2.5, 2.5].
#
# DGPs (from simu_sett.R, originally proposed in Gibbs et al. 2023):
#   Setting 1: Y = 0.5*X + |sin(X)| * N(0,1),  X ~ N(0,1)
#   Setting 2: Y = 0.5*X + 2*dnorm(X,0,1.5) * N(0,1),  X ~ N(0,1)
#
# Width = 2 * score_threshold  (RLCP interval is [Yhat +/- score_threshold])
#
# Usage (from project root):
#   Rscript exp_gibbs_compare/run_rlcp_gibbs.R [--nrep 20] [--h 0.05]

suppressPackageStartupMessages({
  library(mvtnorm)
  library(MASS)
})

# ---- parse simple args ----------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
nrep_arg <- 20
h_arg    <- 0.05
for (i in seq_along(args)) {
  if (args[i] == "--nrep" && i < length(args)) nrep_arg <- as.integer(args[i+1])
  if (args[i] == "--h"    && i < length(args)) h_arg    <- as.numeric(args[i+1])
}

# ---- paths ----------------------------------------------------------------
proj_root <- normalizePath(".", mustWork = TRUE)
utils_dir <- file.path(proj_root, "Conditional_Coverage", "reproduce_rlcp",
                        "RLCP", "utils")
out_dir   <- file.path(proj_root, "exp_gibbs_compare", "output")

source(file.path(utils_dir, "simu_sett.R"))
source(file.path(utils_dir, "methods.R"))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- experiment parameters ------------------------------------------------
nrep   <- nrep_arg
ntrain <- 2000
ncalib <- 2000
ntest  <- 2000
d      <- 1
alpha  <- 0.1
h      <- h_arg

centers <- seq(-2.5, 2.5, by = 0.25)   # 21 points
radius  <- 0.4

cat(sprintf("RLCP Gibbs comparison: nrep=%d, ntrain/ncalib/ntest=%d, h=%.3f\n",
            nrep, ntrain, h))

# ---- helper: one macrorep -------------------------------------------------
one_rep <- function(k, setting) {
  set.seed(k + setting * 1000)

  train_data   <- simulation(ntrain, d, setting)
  Xnames       <- paste0("X", 1:d)
  formula      <- as.formula(paste("Y ~", paste(Xnames, collapse = "+")))
  model_lm     <- lm(formula, data = train_data)

  calib_data   <- simulation(ncalib, d, setting)
  Xcalib       <- as.matrix(calib_data[, -1])   # matrix, shape (ncalib, d)
  scores_calib <- abs(calib_data$Y - predict(model_lm, newdata = calib_data))

  test_data    <- simulation(ntest, d, setting)
  Xtest        <- as.matrix(test_data[, -1])    # matrix, shape (ntest, d)
  scores_test  <- abs(test_data$Y - predict(model_lm, newdata = test_data))

  # RLCP_res: ntest x 2  (col1 = coverage binary, col2 = score_threshold)
  RLCP_res <- RLCP(Xcalib, scores_calib, Xtest, scores_test,
                   "gaussian", h, alpha)

  coverage_pt <- RLCP_res[, 1]         # binary coverage per test point
  width_pt    <- 2 * RLCP_res[, 2]     # width = 2 * score_threshold

  # local average within radius ball around each center
  local_cov   <- numeric(length(centers))
  local_width <- numeric(length(centers))

  Xtest1d <- as.numeric(Xtest)
  for (j in seq_along(centers)) {
    in_ball <- abs(Xtest1d - centers[j]) <= radius
    if (sum(in_ball) == 0) {
      local_cov[j]   <- NA
      local_width[j] <- NA
    } else {
      local_cov[j]   <- mean(coverage_pt[in_ball])
      # exclude Inf (RLCP sets score_threshold=Inf for sparse-region test pts)
      finite_w <- width_pt[in_ball][is.finite(width_pt[in_ball])]
      local_width[j] <- if (length(finite_w) > 0) mean(finite_w) else NA
    }
  }

  list(cov = local_cov, width = local_width,
       n_inf = sum(is.infinite(RLCP_res[, 2])),
       ntest = ntest)
}

# ---- run both settings ----------------------------------------------------
for (setting in 1:2) {
  cat(sprintf("\n--- Setting %d ---\n", setting))

  cov_mat   <- matrix(NA, nrow = nrep, ncol = length(centers))
  width_mat <- matrix(NA, nrow = nrep, ncol = length(centers))

  n_inf_total <- 0
  for (k in 1:nrep) {
    if (k == 1 || k %% 5 == 0) cat(sprintf("  rep %d / %d\n", k, nrep))
    res             <- one_rep(k, setting)
    cov_mat[k, ]   <- res$cov
    width_mat[k, ] <- res$width
    n_inf_total    <- n_inf_total + res$n_inf
  }
  cat(sprintf("  Total Inf score_threshold: %d / %d (%.1f%%)\n",
              n_inf_total, nrep * ntest, 100 * n_inf_total / (nrep * ntest)))

  out <- data.frame(
    center          = centers,
    mean_local_cov  = colMeans(cov_mat,   na.rm = TRUE),
    sd_local_cov    = apply(cov_mat,   2, sd, na.rm = TRUE),
    mean_width      = colMeans(width_mat, na.rm = TRUE),
    sd_width        = apply(width_mat, 2, sd, na.rm = TRUE)
  )

  fname <- file.path(out_dir, sprintf("rlcp_s%d_results.csv", setting))
  write.csv(out, fname, row.names = FALSE)
  cat(sprintf("  Saved -> %s\n", fname))
  cat(sprintf("  Mean coverage: %.4f\n", mean(cov_mat, na.rm = TRUE)))
  cat(sprintf("  Mean width:    %.4f\n", mean(width_mat, na.rm = TRUE)))
}

cat("\nDone.\n")
