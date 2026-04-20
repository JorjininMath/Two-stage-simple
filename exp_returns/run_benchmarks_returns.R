###############################################################################
# run_benchmarks_returns.R
#
# Run DCP benchmark methods on one round of the returns experiment.
# Methods: DCP-QR, DCP-DR, CQR (o/m/r), CP-OLS, CP-loc
#
# Usage (called from Python via subprocess):
#   Rscript exp_returns/run_benchmarks_returns.R data_dir output_csv alpha n_grid
#
# data_dir must contain: X0.csv Y0.csv X1.csv Y1.csv X_test.csv Y_test.csv
###############################################################################

library(quantreg)

# Locate project root via script path, fall back to getwd()
cmd_args    <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", cmd_args[grep("^--file=", cmd_args)])
if (length(script_path) > 0) {
  project_root <- dirname(dirname(normalizePath(script_path[1])))
} else {
  project_root <- getwd()
}
source(file.path(project_root, "dcp_r.R"))

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
load_data <- function(data_dir) {
  read1 <- function(f) as.vector(read.csv(file.path(data_dir, f), header = FALSE)[, 1])
  readM <- function(f) as.matrix(read.csv(file.path(data_dir, f), header = FALSE))
  list(
    X0     = readM("X0.csv"),
    Y0     = read1("Y0.csv"),
    X1     = readM("X1.csv"),
    Y1     = read1("Y1.csv"),
    X_test = readM("X_test.csv"),
    Y_test = read1("Y_test.csv")
  )
}

interval_score <- function(Y, L, U, alpha) {
  (U - L) + (2/alpha) * pmax(0, L - Y) + (2/alpha) * pmax(0, Y - U)
}

# ---------------------------------------------------------------------------
# Run all methods
# ---------------------------------------------------------------------------
run_all <- function(dat, alpha, n_grid) {
  X0 <- dat$X0; Y0 <- dat$Y0
  X1 <- dat$X1; Y1 <- dat$Y1
  X_test <- dat$X_test; Y_test <- dat$Y_test
  n_test <- length(Y_test)

  taus <- seq(0.001, 0.999, length.out = n_grid)
  ys   <- quantile(c(Y0, Y1), seq(0.001, 0.999, length.out = n_grid))

  out <- data.frame(row.names = seq_len(n_test))

  # --- DCP-QR ---
  tryCatch({
    res <- dcp.qr(Y0, X0, Y1, X1, Y_test, X_test, taus, alpha)
    out$covered_qr <- as.integer(res$cov.qr)
    out$L_qr       <- res$lb.qr
    out$U_qr       <- res$ub.qr
    out$width_qr   <- res$leng.qr
    out$IS_qr      <- interval_score(Y_test, res$lb.qr, res$ub.qr, alpha)
  }, error = function(e) message("DCP-QR failed: ", e$message))

  # --- DCP-DR ---
  tryCatch({
    res <- dcp.dr(Y0, X0, Y1, X1, Y_test, X_test, ys, taus, alpha)
    out$covered_dr <- as.integer(res$cov.dr)
    out$L_dr       <- res$lb.dr
    out$U_dr       <- res$ub.dr
    out$width_dr   <- res$leng.dr
    out$IS_dr      <- interval_score(Y_test, res$lb.dr, res$ub.dr, alpha)
  }, error = function(e) message("DCP-DR failed: ", e$message))

  # --- CQR (o / m / r) ---
  tryCatch({
    res <- cqr(Y0, X0, Y1, X1, Y_test, X_test, alpha)
    # CQR-o
    out$covered_cqr  <- as.integer(res$cov.o)
    out$width_cqr    <- res$leng.o
    # CQR-m
    out$covered_cqrm <- as.integer(res$cov.m)
    out$width_cqrm   <- res$leng.m
    # CQR-r
    out$covered_cqrr <- as.integer(res$cov.r)
    out$width_cqrr   <- res$leng.r
  }, error = function(e) message("CQR failed: ", e$message))

  # --- CP-OLS ---
  tryCatch({
    res <- cp.reg(Y0, X0, Y1, X1, Y_test, X_test, alpha)
    out$covered_reg <- as.integer(res$cov.reg)
    out$width_reg   <- res$leng.reg
    out$IS_reg      <- interval_score(Y_test,
                         Y_test - res$leng.reg/2, Y_test + res$leng.reg/2, alpha)
  }, error = function(e) message("CP-OLS failed: ", e$message))

  # --- CP-loc ---
  tryCatch({
    res <- cp.loc(Y0, X0, Y1, X1, Y_test, X_test, alpha)
    out$covered_loc <- as.integer(res$cov.loc)
    out$width_loc   <- res$leng.loc
    out$IS_loc      <- interval_score(Y_test,
                         Y_test - res$leng.loc/2, Y_test + res$leng.loc/2, alpha)
  }, error = function(e) message("CP-loc failed: ", e$message))

  out
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
args     <- commandArgs(trailingOnly = TRUE)
data_dir <- normalizePath(args[1], mustWork = TRUE)
out_csv  <- args[2]
alpha    <- as.numeric(args[3])
n_grid   <- as.integer(args[4])

dat <- load_data(data_dir)
res <- run_all(dat, alpha = alpha, n_grid = n_grid)
dir.create(dirname(out_csv), showWarnings = FALSE, recursive = TRUE)
write.csv(res, out_csv, row.names = FALSE)
