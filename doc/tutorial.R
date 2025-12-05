## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 6,
  fig.height = 4
)
library(ghscenmis)

## ----example-censored-data----------------------------------------------------
set.seed(123)
p <- 10
n <- 200

# True covariance with a 3-variable block
Sigma_true <- diag(p)
Sigma_true[1, 2] <- Sigma_true[2, 1] <-
  Sigma_true[1, 3] <- Sigma_true[3, 1] <-
  Sigma_true[2, 3] <- Sigma_true[3, 2] <- 0.5

Omega_true <- solve(Sigma_true)

# Generate data
Z <- MASS::mvrnorm(n, rep(0, p), Sigma_true)

# Apply left-censoring at variable-specific thresholds
cvec <- rep(c(-0.5, 0.5), length.out = p)
Y <- pmax(Z, matrix(cvec, n, p, byrow = TRUE))

## ----fit-censored-------------------------------------------------------------
fit <- censored_GHS(
  Ytilde = Y,
  cvec = cvec,
  n_iter = 3000,
  burn = 500,
  verbose = FALSE
)

# Posterior mean precision matrix
Omega_hat <- fit$Omega_mean

# Edge selection based on posterior credible intervals
edges <- get_edge_selection_ci(fit$Omega_draws)$E_est

## ----evaluate-censored--------------------------------------------------------
# True adjacency
E_true <- 1 * (abs(Omega_true) > 1e-6)
diag(E_true) <- 0

metrics <- get_edge_metrics(E_true, edges)
metrics

## ----data-missing-------------------------------------------------------------
set.seed(123)
Y <- matrix(rnorm(100 * 5), 100, 5)
Y[sample(length(Y), 10)] <- NA  # introduce missing values

## ----fit-missing--------------------------------------------------------------
fit_mis <- ghs_missing_gibbs(Y = Y)

Omega_mis <- fit_mis$Omega_mean
Omega_mis

