

# **GHScenmis: Sparse Graphical Models with Censored or Missing Data**

`GHScenmis` is an R package for estimating **sparse precision matrices** (graphical models) when your data contain **censored observations** (e.g., left/right censored) or **missing values**. The package implements the **Graphical Horseshoe prior (GHS)** with an efficient **Gibbs sampler** that can handle these data irregularities.

---

## **Installation**

You can install the development version directly from GitHub using `devtools`:

```r
# install.packages("devtools") # if not installed
devtools::install_github("tienmt/ghscenmis")
library(ghscenmis)
```
or install with vignettes "tutorial"
```r
devtools::install_github(
    "tienmt/ghscenmis", 
    build_vignettes = TRUE,
    build_opts = c("--no-resave-data", "--no-manual")
)
```

---

## **Example 1: Sparse Graphical Model with Censored Data**

This example demonstrates fitting a sparse precision matrix when some entries are **left-censored**.

```r
set.seed(123)
p <- 10
n <- 200

# --- True covariance matrix with a 3-node block structure ---
Sigma_true <- diag(p)
Sigma_true[1, 2] <- Sigma_true[2, 1] <-
  Sigma_true[1, 3] <- Sigma_true[3, 1] <-
  Sigma_true[2, 3] <- Sigma_true[3, 2] <- 0.5

# True precision matrix
Omega_true <- solve(Sigma_true)

# True edge adjacency matrix
E_true <- 1 * (abs(Omega_true) > 1e-6)
diag(E_true) <- 0

# --- Generate data ---
Z <- MASS::mvrnorm(n, rep(0, p), Sigma_true)

# Apply variable-wise left-censoring at thresholds
cvec <- rep(c(-0.5, 0.5), length.out = p)
Y <- pmax(Z, matrix(cvec, n, p, byrow = TRUE))

# --- Fit censored Graphical Horseshoe ---
fit <- censored_GHS(
  Ytilde = Y,
  cvec = cvec,
  n_iter = 6000,
  burn = 1000,
  verbose = FALSE
)

# Posterior mean precision matrix
Omega_hat <- fit$Omega_mean

# Edge selection based on posterior credible intervals
E_est <- get_edge_selection_ci(fit$Omega_draws)$E_est

# Evaluate estimation and edge selection accuracy
c(
  MSE = sum((Omega_hat - Omega_true)^2),
  get_edge_metrics(E_true, E_est)
)
```

**Explanation:**

* `Ytilde`: observed censored data
* `cvec`: censoring thresholds per variable
* `fit$Omega_mean`: posterior mean of the precision matrix
* `E_est`: estimated adjacency matrix using credible intervals
* `get_edge_metrics()`: computes standard edge selection metrics (TPR, FPR, Precision, F1)

---

## **Example 2: Sparse Graphical Model with Missing Data**

`GHScenmis` can also handle missing values seamlessly using a **data-augmentation Gibbs sampler**.

```r
set.seed(123)
Y <- matrix(rnorm(100 * 5), 100, 5)
Y[sample(length(Y), 10)] <- NA  # randomly introduce missing values

# Fit the Graphical Horseshoe with missing data
fit <- ghs_missing_gibbs(Y = Y)

# Posterior mean precision matrix
fit$Omega_mean
```

**Notes:**

* `NA` values are automatically imputed within the Gibbs sampler.
* `fit$Omega_mean` gives the posterior mean of the precision matrix, which can be used for downstream inference or edge selection.

---

## **Key Functions**

| Function                  | Purpose                                                                 |
| ------------------------- | ----------------------------------------------------------------------- |
| `censored_GHS()`          | Fits a sparse graphical model when data are censored.                   |
| `ghs_missing_gibbs()`     | Fits a sparse graphical model when data have missing values.            |
| `get_edge_selection_ci()` | Computes posterior credible intervals and returns a binary edge matrix. |
| `get_edge_metrics()`      | Computes standard edge selection metrics: TPR, FPR, Precision, F1.      |

---

## **References**

* Li, Y., Craig, B. A., and Bhadra, A. (2019). The graphical horseshoe estimator for inverse
covariance matrices. Journal of Computational and Graphical Statistics, 28(3):747–757.*.

---


