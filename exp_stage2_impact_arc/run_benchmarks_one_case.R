library(hetGP)
library(quantreg)

cmd_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", cmd_args[grep("^--file=", cmd_args)])
if (length(script_path) > 0) {
  project_root <- dirname(dirname(normalizePath(script_path[1])))
} else {
  project_root <- getwd()
}
source(file.path(project_root, "dcp_r.R"))

load_case_data <- function(data_dir) {
  rep_dir <- file.path(data_dir, "macrorep_0")
  X0 <- as.matrix(read.csv(file.path(rep_dir, "X0.csv"), header = FALSE))
  Y0 <- as.vector(read.csv(file.path(rep_dir, "Y0.csv"), header = FALSE)[, 1])
  X1 <- as.matrix(read.csv(file.path(rep_dir, "X1.csv"), header = FALSE))
  Y1 <- as.vector(read.csv(file.path(rep_dir, "Y1.csv"), header = FALSE)[, 1])
  X_test <- as.matrix(read.csv(file.path(rep_dir, "X_test.csv"), header = FALSE))
  Y_test <- as.vector(read.csv(file.path(rep_dir, "Y_test.csv"), header = FALSE)[, 1])
  list(X0 = X0, Y0 = Y0, X1 = X1, Y1 = Y1, X_test = X_test, Y_test = Y_test)
}

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

  dcp_dr <- dcp.dr(Y0, X0, Y1, X1, Y_test, X_test, ys, taus, alpha_sig)
  L_dr <- dcp_dr$lb.dr
  U_dr <- dcp_dr$ub.dr
  width_dr <- dcp_dr$leng.dr
  covered_interval_dr <- as.integer((Y_test >= L_dr) & (Y_test <= U_dr))
  covered_score_dr <- as.integer(dcp_dr$cov.dr)
  interval_score_dr <- width_dr +
    (2/alpha) * pmax(0, L_dr - Y_test) +
    (2/alpha) * pmax(0, Y_test - U_dr)

  X_hetgp <- rbind(X0, X1)
  Z_hetgp <- c(Y0, Y1)
  prdata <- find_reps(X_hetgp, Z_hetgp, rescale = FALSE, normalize = FALSE)
  hetgp_model <- mleHetGP(
    X = list(X0 = prdata$X0, Z0 = prdata$Z0, mult = prdata$mult),
    Z = prdata$Z,
    lower = rep(0.01, d),
    upper = rep(10, d),
    covtype = "Gaussian"
  )
  hetgp_pred <- predict(x = X_test, object = hetgp_model)
  z_alpha <- qnorm(1 - alpha/2)
  hetgp_L <- hetgp_pred$mean - z_alpha * sqrt(hetgp_pred$sd2 + hetgp_pred$nugs)
  hetgp_U <- hetgp_pred$mean + z_alpha * sqrt(hetgp_pred$sd2 + hetgp_pred$nugs)
  hetgp_width <- hetgp_U - hetgp_L
  hetgp_covered <- as.integer((Y_test >= hetgp_L) & (Y_test <= hetgp_U))
  hetgp_interval_score <- hetgp_width +
    (2/alpha) * pmax(0, hetgp_L - Y_test) +
    (2/alpha) * pmax(0, Y_test - hetgp_U)

  data.frame(
    L_dr = L_dr,
    U_dr = U_dr,
    covered_interval_dr = covered_interval_dr,
    covered_score_dr = covered_score_dr,
    width_dr = width_dr,
    interval_score_dr = interval_score_dr,
    L_hetgp = hetgp_L,
    U_hetgp = hetgp_U,
    covered_interval_hetgp = hetgp_covered,
    width_hetgp = hetgp_width,
    interval_score_hetgp = hetgp_interval_score,
    stringsAsFactors = FALSE
  )
}

args <- commandArgs(trailingOnly = TRUE)
data_dir <- normalizePath(args[1], mustWork = TRUE)
output_csv <- args[2]
alpha <- as.numeric(args[3])
n_grid <- as.integer(args[4])

dat <- load_case_data(data_dir)
out <- run_benchmarks_one_case(dat, alpha = alpha, n_grid = n_grid)
dir.create(dirname(output_csv), showWarnings = FALSE, recursive = TRUE)
write.csv(out, output_csv, row.names = FALSE)
