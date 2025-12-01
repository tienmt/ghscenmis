

#' Gibbs Sampler for Gaussian Graphical Model with Missing Data and Horseshoe Prior
#'
#' Performs Bayesian estimation of a precision matrix for a Gaussian Graphical Model (GGM)
#' when the data contains missing entries. Missing values are imputed using the
#' conditional multivariate normal distribution, and node-wise regressions are
#' regularized with the horseshoe prior.
#'
#' @param Y Numeric matrix of size n x p with observations; may contain NAs.
#' @param n_iter Integer. Total number of Gibbs sampler iterations. Default 3000.
#' @param burnin Integer. Number of burn-in iterations to discard. Default 1000.
#' @param thin Integer. Thinning interval for storage. Default 5.
#' @param eta Numeric. Small regularization parameter for numerical stability. Default 1e-2.
#' @param verbose Logical. Print progress messages during sampling. Default TRUE.
#' @param store_Z Logical. If TRUE, stores sampled latent Z matrices. Default FALSE.
#' @param seed Integer. Random seed for reproducibility. Default NULL.
#' @param a0 Numeric. Shape parameter for inverse-gamma prior on residual variances. Default 1e-2.
#' @param b0 Numeric. Rate parameter for inverse-gamma prior on residual variances. Default 1e-2.
#' @param symmetrize Character. Method to symmetrize the precision matrix: "average", "min", or "max". Default "average".
#'
#' @return A list containing:
#' \describe{
#'   \item{Omega_draws}{Array of sampled precision matrices, dimension p x p x n_keep.}
#'   \item{Omega_mean}{Posterior mean of Omega.}
#'   \item{Omega_median}{Posterior median of Omega.}
#'   \item{Z_draws}{Array of imputed latent Z matrices (if `store_Z = TRUE`).}
#'   \item{settings}{List of sampler settings.}
#' }
#'
#' @details
#' The function implements a Gibbs sampler with the following steps:
#' \enumerate{
#'   \item Impute missing values in Y using the conditional multivariate normal.
#'   \item Perform node-wise linear regressions of each variable on all others,
#'         using the horseshoe prior for sparsity regularization.
#'   \item Reconstruct the precision matrix Omega from the regression coefficients.
#'   \item Symmetrize Omega according to the chosen method, and optionally project to SPD.
#' }
#' Local and global horseshoe parameters are updated using standard conjugate Gibbs steps.
#'
#' @references
#' Makalic, E., & Schmidt, D. F. (2016). A simple sampler for the horseshoe estimator.
#' \emph{IEEE Signal Processing Letters}, 23(1), 179-182.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' Y <- matrix(rnorm(100*5), 100, 5)
#' Y[sample(length(Y), 10)] <- NA
#' fit <- ghs_missing_gibbs( Y, n_iter = 100, burnin = 50, thin = 2)
#' fit$Omega_mean
#'
#' }
#' @export
#' @useDynLib ghscenmis, .registration = TRUE
#' @importFrom Rcpp sourceCpp
ghs_missing_gibbs <- function(Y, n_iter = 3000, burnin = 1000, thin = 1, eta = 1e-2,
                              verbose = TRUE, store_Z = FALSE, seed = NULL,
                              a0 = 1e-2, b0 = 1e-2,
                              symmetrize = "average"
                              ) {
  ghs_missing_gibbs_cpp( Y, n_iter, burnin, thin, verbose,
        store_Z, a0, b0, symmetrize)

  }







#' Gibbs Sampler for Censored Graphical Horseshoe Precision Matrix Estimation
#'
#' @description
#' Implements a Gibbs sampler for estimating a sparse precision matrix
#' under the **Graphical Horseshoe (GHS)** prior when the data are
#' **censored** (left- or right-censored). The sampler is fully implemented in C++
#' for computational efficiency.
#'
#' Given observations \eqn{Y^{\mathrm{tilde}} \in \mathbb{R}^{n \times p}} where
#' each variable may be censored at a known threshold, the algorithm alternates
#' between:
#' \itemize{
#'   \item Sampling latent uncensored values from truncated normals.
#'   \item Updating the precision matrix \eqn{\Omega} using Graphical Horseshoe shrinkage.
#' }
#'
#' @param Ytilde A numeric matrix of dimension \eqn{n \times p} containing the **observed**
#'   values after censoring. For left censoring, values satisfy \eqn{Y_{ij} = \max(Z_{ij}, c_j)}.
#' @param cvec A numeric vector of length \eqn{p} giving censoring thresholds for each variable.
#'   Interpreted as **left**-censoring by default; use `right_or_left` to switch.
#' @param n_iter Total number of MCMC iterations.
#' @param burn Number of burn-in iterations to discard.
#' @param thin Thinning interval for storing posterior draws.
#' @param a0 Hyperparameters for the global horseshoe scale parameter.
#' @param b0 Hyperparameters for the global horseshoe scale parameter.
#' @param symmetrize Method used to enforce symmetry of sampled precision matrices.
#'   One of `"average"` (default), `"lower"`, or `"upper"`.
#' @param project_spd Logical; if `TRUE`, enforces positive-definiteness via nearest SPD projection.
#' @param verbose Logical; if `TRUE`, prints progress during sampling.
#' @param right_or_left `"left"` (default) for left-censored data or `"right"` for right-censoring.
#'
#' @return A list containing:
#' \describe{
#'   \item{Omega_mean}{Posterior mean of the precision matrix (after burn-in and thinning).}
#'   \item{Omega_draws}{Array of dimension \eqn{p \times p \times m} containing posterior draws,
#'     where \eqn{m = (n\_iter - burn)/thin}.}
#'   \item{Sigma_mean}{Posterior mean of the covariance matrix.}
#'   \item{latent_Z}{Posterior draws of the imputed latent uncensored data.}
#'   \item{tau2_global}{Posterior draws of the global horseshoe scale parameter.}
#' }
#'
#' @details
#' This function implements a censored-data extension of the Graphical Horseshoe
#' sampler from li2019graphical GHS.
#' Latent Gaussian variables are sampled from truncated normal distributions
#' conditional on the censoring mechanism. The precision matrix is updated using
#' fully conditional Gaussian–Horseshoe steps.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' p <- 10
#' n <- 200
#'
#' # True covariance with a 3-node block structure
#' Sigma_true <- diag(p)
#' Sigma_true[1, 2] <- Sigma_true[2, 1] <-
#'   Sigma_true[1, 3] <- Sigma_true[3, 1] <-
#'   Sigma_true[2, 3] <- Sigma_true[3, 2] <- 0.5
#'
#' Omega_true <- solve(Sigma_true)
#' E_true <- 1 * (abs(Omega_true) > 1e-6)
#' diag(E_true) <- 0
#'
#' # Generate data
#' Z <- MASS::mvrnorm(n, rep(0, p), Sigma_true)
#'
#' # Apply variable-wise left-censoring at thresholds
#' cvec <- rep(c(-0.5, 0.5), length.out = p)
#' Y <- pmax(Z, matrix(cvec, n, p, byrow = TRUE))
#'
#' # Fit censored Graphical Horseshoe
#' fit <- censored_GHS(
#'   Ytilde = Y,
#'   cvec = cvec,
#'   n_iter = 6000,
#'   burn = 1000,
#'   verbose = FALSE
#' )
#'
#' # Posterior mean precision
#' Omega_hat <- fit$Omega_mean
#'
#' # Edge selection
#' E_est <- get_edge_selection_ci(fit$Omega_draws)$E_est
#'
#' # Evaluate accuracy
#' c(
#'   MSE = sum((Omega_hat - Omega_true)^2),
#'   get_edge_metrics(E_true, E_est)
#' )
#' }
#'
#'
#' @export
#' @useDynLib ghscenmis, .registration = TRUE
censored_GHS <- function(Ytilde, cvec, n_iter = 2000L, burn = 1000L,
                                    thin = 1L, a0 = 1e-2, b0 = 1e-2,
                                    symmetrize = "average", project_spd = FALSE,
                                    verbose = TRUE, right_or_left = "left") {
  gibbs_censored_GHS_cpp2( Ytilde, cvec, n_iter, burn, thin,
        a0, b0, symmetrize, project_spd, verbose, right_or_left)
}










#' Edge Selection via Posterior Credible Intervals
#'
#' @description
#' Computes posterior credible intervals for each entry of a precision matrix
#' sampled using MCMC, and produces a binary edge-selection matrix based on
#' whether the credible interval excludes zero.
#'
#' @param Omega_draws A 3-dimensional array of dimension
#'   \eqn{p \times p \times M}, where \eqn{M} is the number of posterior draws
#'   of the precision matrix \eqn{\Omega}. Each slice \code{Omega_draws[,,m]}
#'   contains the \eqn{p \times p} precision matrix from iteration \eqn{m}.
#' @param level Credible interval level (default \code{0.95}). The function
#'   computes marginal credible intervals for each entry based on the empirical
#'   posterior quantiles.
#'
#' @details
#' For each pair \eqn{(i, j)}, the function extracts the posterior samples
#' \eqn{\{\Omega_{ij}^{(m)}\}_{m=1}^M} and computes the lower and upper
#' empirical quantiles corresponding to a \code{level}-credible interval.
#'
#' An edge between nodes \eqn{i} and \eqn{j} is considered present if the
#' credible interval excludes zero:
#'
#' \deqn{
#'   \text{Edge selected if } \mathrm{CI}_{ij}^{\text{low}} \cdot
#'   \mathrm{CI}_{ij}^{\text{high}} > 0.
#' }
#'
#' That is, both bounds are either positive or negative. The diagonal entries
#' are always set to zero in the returned edge matrix.
#'
#' @return
#' A list with the following components:
#' \describe{
#'   \item{\code{E_est}}{A \eqn{p \times p} binary matrix indicating edge
#'     selection based on credible intervals.}
#'   \item{\code{ci_low}}{A \eqn{p \times p} matrix of lower credible bounds.}
#'   \item{\code{ci_high}}{A \eqn{p \times p} matrix of upper credible bounds.}
#' }
#'
#' @export
#' @importFrom stats quantile
get_edge_selection_ci <- function(Omega_draws, level = 0.95) {
  p <- dim(Omega_draws)[1]
  ci_low <- ci_high <- matrix(0, p, p)
  E_est <- matrix(0, p, p)
  alpha <- (1 - level) / 2
  lower_q <- alpha
  upper_q <- 1 - alpha

  for (i in 1:p) {
    for (j in 1:p) {
      post_samples <- Omega_draws[i, j, ]
      ci <- quantile(post_samples, probs = c(lower_q, upper_q))
      ci_low[i, j]  <- ci[1]
      ci_high[i, j] <- ci[2]
      # Edge selected if CI excludes zero
      if (ci[1] * ci[2] > 0) {
        E_est[i, j] <- 1
      }
    }
  }
  diag(E_est) <- 0
  list(E_est = E_est, ci_low = ci_low, ci_high = ci_high)
}








#' Compute Edge Selection Performance Metrics
#'
#' @description
#' Computes standard evaluation metrics for edge selection in graphical models.
#' Given a true edge adjacency matrix and an estimated one (both symmetric
#' \eqn{p \times p} matrices with zeros on the diagonal), the function evaluates
#' performance using only the upper triangular entries to avoid double-counting.
#'
#' @param E_true A \eqn{p \times p} binary matrix indicating the true edge
#'   structure of the graph. Diagonal entries must be zero.
#' @param E_est A \eqn{p \times p} binary matrix indicating the estimated edge
#'   structure. Must have the same dimensions as \code{E_true}.
#'
#' @details
#' The function computes:
#' \itemize{
#'   \item \strong{TPR} — True Positive Rate (Recall):
#'     \eqn{TP / (TP + FN)}.
#'   \item \strong{FPR} — False Positive Rate:
#'     \eqn{FP / (FP + TN)}.
#'   \item \strong{Precision} — Proportion of selected edges that are correct:
#'     \eqn{TP / (TP + FP)}.
#'   \item \strong{F1 Score} — Harmonic mean of Precision and TPR.
#' }
#'
#' Edges are counted only once by restricting to the upper-triangular part of
#' the matrices.
#'
#' @return
#' A named numeric vector containing:
#' \describe{
#'   \item{TPR}{True Positive Rate}
#'   \item{FPR}{False Positive Rate}
#'   \item{Precision}{Precision}
#'   \item{F1}{F1 score}
#' }
#'
#' @export
get_edge_metrics <- function(E_true, E_est) {
  upper <- upper.tri(E_true)
  TP <- sum(E_true[upper] == 1 & E_est[upper] == 1)
  FP <- sum(E_true[upper] == 0 & E_est[upper] == 1)
  FN <- sum(E_true[upper] == 1 & E_est[upper] == 0)
  TN <- sum(E_true[upper] == 0 & E_est[upper] == 0)

  TPR <- TP / (TP + FN)
  FPR <- FP / (FP + TN)
  Precision <- TP / (TP + FP)
  F1 <- 2 * Precision * TPR / (Precision + TPR)

  c(TPR = TPR, FPR = FPR, Precision = Precision, F1 = F1)
}

