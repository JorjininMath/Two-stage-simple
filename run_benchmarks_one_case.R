#############################################################################
# Run DCP-DR and hetGP for one case (one macrorep's data in one directory).
# Reads: data_dir/macrorep_0/{X0,Y0,X1,Y1,X_test,Y_test}.csv
# Writes: one CSV with per-point L_dr, U_dr, L_hetgp, U_hetgp, etc.
#
# Usage: Rscript run_benchmarks_one_case.R <data_dir> <output_csv> <alpha> <n_grid>
#############################################################################

library(hetGP)
library(quantreg)

# Project root = directory containing this script
cmd_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", cmd_args[grep("^--file=", cmd_args)])
if (length(script_path) > 0) {
  project_root <- dirname(normalizePath(script_path[1]))
} else {
  project_root <- getwd()
}
source(file.path(project_root, "dcp_r.R"))

# ============================================================================
# Load one case (macrorep_0 from data_dir)
# ============================================================================
load_case_data <- function(data_dir) {
  rep_dir <- file.path(data_dir, "macrorep_0")
  if (!dir.exists(rep_dir)) stop(sprintf("Directory not found: %s", rep_dir))
  X0 <- as.matrix(read.csv(file.path(rep_dir, "X0.csv"), header = FALSE))
  Y0 <- as.vector(read.csv(file.path(rep_dir, "Y0.csv"), header = FALSE)[, 1])
  X1 <- as.matrix(read.csv(file.path(rep_dir, "X1.csv"), header = FALSE))
  Y1 <- as.vector(read.csv(file.path(rep_dir, "Y1.csv"), header = FALSE)[, 1])
  X_test <- as.matrix(read.csv(file.path(rep_dir, "X_test.csv"), header = FALSE))
  Y_test <- as.vector(read.csv(file.path(rep_dir, "Y_test.csv"), header = FALSE)[, 1])
  list(X0 = X0, Y0 = Y0, X1 = X1, Y1 = Y1, X_test = X_test, Y_test = Y_test)
}

# ============================================================================
# Run DCP-DR and hetGP, return per-point results as data frame
# ============================================================================
run_benchmarks_one_case <- function(dat, alpha = 0.1, n_grid = 500) {
  X0 <- dat$X0
  Y0 <- dat$Y0
  X1 <- dat$X1
  Y1 <- dat$Y1
  X_test <- dat$X_test
  Y_test <- dat$Y_test

  n_test <- length(Y_test)
  d <- ncol(X0)
  alpha_sig <- alpha
  taus <- seq(0.01, 0.99, length.out = n_grid)
  ys <- seq(min(Y0), max(Y0), length.out = n_grid)

  # --- DCP-DR ---
  dcp_dr <- dcp.dr(Y0, X0, Y1, X1, Y_test, X_test, ys, taus, alpha_sig)
  L_dr <- dcp_dr$lb.dr
  U_dr <- dcp_dr$ub.dr
  width_dr <- dcp_dr$leng.dr
  covered_interval_dr <- as.integer((Y_test >= L_dr) & (Y_test <= U_dr))
  covered_score_dr <- as.integer(dcp_dr$cov.dr)
  interval_score_dr <- width_dr +
    (2 / alpha) * pmax(0, L_dr - Y_test) +
    (2 / alpha) * pmax(0, Y_test - U_dr)

  # --- DCP-QR ---
  dcp_qr <- dcp.qr(Y0, X0, Y1, X1, Y_test, X_test, taus, alpha_sig)
  L_qr <- dcp_qr$lb.qr
  U_qr <- dcp_qr$ub.qr
  width_qr <- dcp_qr$leng.qr
  covered_score_qr <- as.integer(dcp_qr$cov.qr)
  covered_interval_qr <- as.integer((Y_test >= L_qr) & (Y_test <= U_qr))
  interval_score_qr <- width_qr +
    (2 / alpha) * pmax(0, L_qr - Y_test) +
    (2 / alpha) * pmax(0, Y_test - U_qr)

  # --- hetGP ---
  hetgp_L <- hetgp_U <- hetgp_width <- rep(NA_real_, n_test)
  hetgp_covered <- rep(NA_integer_, n_test)
  hetgp_interval_score <- rep(NA_real_, n_test)
  hetgp_covered_score <- rep(NA_integer_, n_test)
  tryCatch({
    X_train <- rbind(X0, X1)
    Y_train <- c(Y0, Y1)
    reps <- find_reps(X_train, Y_train)
    hetgp_model <- mleHetGP(
      X = list(X0 = reps$X0, Z0 = reps$Z0, mult = reps$mult),
      Z = reps$Z,
      lower = rep(0.01, ncol(X0)),
      upper = rep(10, ncol(X0)),
      covtype = "Gaussian",
      noiseControl = list(g_min = 1e-4, g_bounds = c(1e-4, 100))
    )
    pred_hetgp <- predict(hetgp_model, X_test)
    z_alpha <- qnorm(1 - alpha_sig / 2)
    total_var <- pred_hetgp$sd2 + pred_hetgp$nugs
    hetgp_L <- pred_hetgp$mean - z_alpha * sqrt(total_var)
    hetgp_U <- pred_hetgp$mean + z_alpha * sqrt(total_var)
    hetgp_width <- hetgp_U - hetgp_L
    hetgp_covered <- as.integer((Y_test >= hetgp_L) & (Y_test <= hetgp_U))
    hetgp_covered_score <- hetgp_covered
    hetgp_interval_score <- hetgp_width +
      (2 / alpha) * pmax(0, hetgp_L - Y_test) +
      (2 / alpha) * pmax(0, Y_test - hetgp_U)
  }, error = function(e) {
    warning(sprintf("hetGP failed: %s", conditionMessage(e)))
  })

  data.frame(
    L_dr                = L_dr,
    U_dr                = U_dr,
    covered_interval_dr = covered_interval_dr,
    covered_score_dr    = covered_score_dr,
    width_dr            = width_dr,
    interval_score_dr   = interval_score_dr,
    L_qr                   = L_qr,
    U_qr                   = U_qr,
    covered_interval_qr    = covered_interval_qr,
    covered_score_qr       = covered_score_qr,
    width_qr               = width_qr,
    interval_score_qr      = interval_score_qr,
    L_hetgp                = hetgp_L,
    U_hetgp                = hetgp_U,
    covered_interval_hetgp = hetgp_covered,
    covered_score_hetgp    = hetgp_covered_score,
    width_hetgp            = hetgp_width,
    interval_score_hetgp   = hetgp_interval_score,
    stringsAsFactors = FALSE
  )
}

# ============================================================================
# CLI
# ============================================================================
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  cat("Usage: Rscript run_benchmarks_one_case.R <data_dir> <output_csv> <alpha> <n_grid>\n")
  quit(status = 1)
}

data_dir   <- normalizePath(args[1], mustWork = TRUE)
output_csv <- args[2]
alpha      <- as.numeric(args[3])
n_grid     <- as.integer(args[4])

dat <- load_case_data(data_dir)
out <- run_benchmarks_one_case(dat, alpha = alpha, n_grid = n_grid)
dir.create(dirname(output_csv), showWarnings = FALSE, recursive = TRUE)
write.csv(out, output_csv, row.names = FALSE)
