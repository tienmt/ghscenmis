// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// ----------------- helpers -----------------

// Symmetrize matrix: 0.5 * (M + M')
arma::mat symmetrize_mat(const arma::mat& M) {
  return 0.5 * (M + M.t());
}

// Safe pseudo-inverse
arma::mat pinv_safe(const arma::mat& A) {
  return arma::pinv(A);
}

// Robust inverse for positive definite matrices
// Tries sympd, falls back to pinv if singular
arma::mat inv_safe_sympd(const arma::mat& A) {
  arma::mat B;
  bool status = arma::inv_sympd(B, A);
  if (!status) {
    B = arma::pinv(A);
  }
  return B;
}

// Sample multivariate normal with covariance Sigma
arma::vec rmvnorm(const arma::vec& mu, const arma::mat& Sigma) {
  int d = mu.n_rows;
  if (d == 1) {
    double var = Sigma(0,0);
    double sd = (var > 0) ? std::sqrt(var) : 0.0;
    arma::vec out(1);
    out(0) = mu(0) + R::rnorm(0.0, sd);
    return out;
    }

  // Use eigen decomposition for stability
  arma::vec eigval;
  arma::mat eigvec;
  bool ok = arma::eig_sym(eigval, eigvec, Sigma);

  if (!ok) {
    // Fallback if eig_sym fails
    return mu + arma::randn<arma::vec>(d);
  }

  // Ensure non-negative eigenvalues
  eigval.elem(find(eigval < 0)).zeros();

  arma::mat R_mat = eigvec * arma::diagmat(arma::sqrt(eigval));
  arma::vec z = arma::randn<arma::vec>(d);
  return mu + R_mat * z;
}

// Impute missing values rowwise using conditional multivariate normal
arma::mat impute_missing_arma(const arma::mat& Y_orig, arma::mat Z, const arma::mat& Sigma) {
  int n = Y_orig.n_rows;
  int p = Y_orig.n_cols;

  for (int i = 0; i < n; ++i) {
    // Check which cols are finite for this row
    arma::uvec indices = arma::regspace<arma::uvec>(0, p-1);
    arma::rowvec row_i = Y_orig.row(i);

    // arma::find_finite works on vectors/matrices and returns linear indices.
    // Since we extract a row, we treat it as a vector.
    arma::uvec obs_idx = arma::find_finite(row_i);
    arma::uvec mis_idx = arma::find_nonfinite(row_i);

    if (mis_idx.n_elem == 0) continue;

    // Build submatrices
    arma::mat Sigma_oo = Sigma.submat(obs_idx, obs_idx);
    arma::mat Sigma_mo = Sigma.submat(mis_idx, obs_idx);
    arma::mat Sigma_om = Sigma.submat(obs_idx, mis_idx);
    arma::mat Sigma_mm = Sigma.submat(mis_idx, mis_idx);

    arma::vec z_o = Z.submat(arma::uvec{ (arma::uword)i }, obs_idx).t();

    // Compute conditional mean: mu_star = Sigma_mo * inv(Sigma_oo) * z_o
    // Use robust solve instead of explicit inv for stability
    arma::mat inv_Sigma_oo = inv_safe_sympd(Sigma_oo);
    arma::vec mu_star = Sigma_mo * (inv_Sigma_oo * z_o);

    // Compute Schur complement (conditional covariance)
    arma::mat Schur = Sigma_mm - Sigma_mo * (inv_Sigma_oo * Sigma_om);
    Schur = symmetrize_mat(Schur);

    // Sample missing values
    arma::vec z_m = rmvnorm(mu_star, Schur);

    // Fill Z
    Z.submat(arma::uvec{ (arma::uword)i }, mis_idx) = z_m.t();
  }
  return Z;
}

// Project symmetric matrix to nearest positive semidefinite
arma::mat nearest_spd(const arma::mat& A) {
  arma::mat B = symmetrize_mat(A);
  arma::vec eigval;
  arma::mat eigvec;
  arma::eig_sym(eigval, eigvec, B);

  // Clamp negative eigenvalues to a small epsilon or 0
  eigval.elem(find(eigval < 0)).zeros();

  arma::mat Bpos = eigvec * arma::diagmat(eigval) * eigvec.t();
  return symmetrize_mat(Bpos);
}

// Compute elementwise median of cube (p x p x n_keep)
arma::mat cube_median(const arma::cube& C) {
  int p = C.n_rows;
  int n_keep = C.n_slices;
  arma::mat M(p, p);

  // Helper vector for sorting
  std::vector<double> tmp(n_keep);

  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      for (int s = 0; s < n_keep; ++s) {
        tmp[s] = C(i,j,s);
      }
      std::sort(tmp.begin(), tmp.end());

      if (n_keep % 2 == 1) {
        M(i,j) = tmp[n_keep / 2];
      } else {
        M(i,j) = 0.5 * (tmp[n_keep / 2 - 1] + tmp[n_keep / 2]);
      }
    }
  }
  return M;
}

// ----------------- Nodewise Step -----------------
// NOTE: Changed to pass std::vector references for speed.
// Avoids converting R Lists <-> C++ objects every iteration.

void nodewise_horseshoe_update(const arma::mat& Z,
                               arma::vec& tau2,
                               std::vector<arma::vec>& lambda2_list,
                               arma::vec& sigma2,
                               std::vector<arma::vec>& nu_list,
                               arma::vec& xi,
                               std::vector<arma::vec>& theta_list,
                               int n, int p,
                               double a0, double b0) {

  for (int j = 0; j < p; j++) {
    arma::vec y = Z.col(j);
    arma::mat X = Z;
    X.shed_col(j); // drop column j
    int m = X.n_cols;

    arma::mat XtX = X.t() * X;
    arma::vec Xty = X.t() * y;

    // Reference to current lambda vector
    arma::vec& lambda2_j = lambda2_list[j];

    // A = XtX + diag(1 / (tau2 * lambda2))
    arma::mat A = XtX;
    double tau2_j = tau2[j];
    for(int k=0; k<m; ++k) {
      A(k,k) += 1.0 / (tau2_j * lambda2_j[k]);
    }

    // V = inv(A)
    arma::mat V = inv_safe_sympd(A);
    arma::vec mu_theta = V * Xty;

    // Sample theta ~ N(mu_theta, sigma2[j] * V)
    // Fix: Use 2-arg chol to catch errors, or just use rmvnorm logic
    arma::mat L_cov = sigma2[j] * V;
    arma::mat L;
    bool chol_ok = arma::chol(L, L_cov, "lower");

    arma::vec theta(m);
    arma::vec noise = arma::randn<arma::vec>(m);

    if (chol_ok) {
      theta = mu_theta + L * noise;
    } else {
      // Fallback if numerical issues in Cholesky
      theta = rmvnorm(mu_theta, L_cov);
    }

    theta_list[j] = theta;

    // Update sigma2[j]
    arma::vec res = y - X * theta;
    double ssq = arma::dot(res, res);
    double rate_sigma = b0 + 0.5 * ssq;
    sigma2[j] = 1.0 / R::rgamma(a0 + n / 2.0, 1.0 / rate_sigma);

    // Update lambda2
    arma::vec& nu_j = nu_list[j]; // ref
    arma::vec lambda2_new(m);

    for (int k = 0; k < m; k++) {
      double th2_scaled = (theta[k] * theta[k]) / (2.0 * sigma2[j] * tau2[j]);
      double rate_lam = (1.0 / nu_j[k]) + th2_scaled;
      lambda2_new[k] = 1.0 / R::rgamma(1.0, 1.0 / rate_lam);
    }
    lambda2_list[j] = lambda2_new;

    // Update nu
    arma::vec nu_new(m);
    for (int k = 0; k < m; k++) {
      nu_new[k] = 1.0 / R::rgamma(1.0, 1.0 / (1.0 + 1.0 / lambda2_new[k]));
    }
    nu_list[j] = nu_new;

    // Update tau2[j]
    double sum_term = 0.0;
    for(int k=0; k<m; ++k) {
      sum_term += (theta[k] * theta[k]) / lambda2_new[k];
    }
    double rate_tau = (1.0 / xi[j]) + 0.5 * sum_term / sigma2[j];
    tau2[j] = 1.0 / R::rgamma((m + 1.0) / 2.0, 1.0 / rate_tau);

    // Update xi[j]
    xi[j] = 1.0 / R::rgamma(1.0, 1.0 / (1.0 + 1.0 / tau2[j]));
  }
}

// ----------------- main exported function -----------------

// [[Rcpp::export]]
List ghs_missing_gibbs_cpp(const arma::mat& Y,
                           int n_iter = 3000,
                           int burnin = 1000,
                           int thin = 5,
                           bool verbose = true,
                           bool store_Z = false,
                           double a0 = 1e-2,
                           double b0 = 1e-2,
                           std::string symmetrize = "average",
                           bool project_spd = false) {

  int n = Y.n_rows;
  int p = Y.n_cols;

  // --- Initialization of Z ---
  arma::mat Z = Y;

  // Simple imputation for initialization: fill NA with column mean
  for (int j = 0; j < p; ++j) {
    arma::vec col_j = Y.col(j);
    arma::uvec finite_idx = arma::find_finite(col_j);

    double mean_val = 0.0;
    if (finite_idx.n_elem > 0) {
      mean_val = arma::mean(col_j.elem(finite_idx));
    }

    arma::uvec nonfinite_idx = arma::find_nonfinite(col_j);
    Z.elem(nonfinite_idx + j * n).fill(mean_val);
  }

  // --- Initial Omega ---
  arma::mat Zc = Z;
  Zc.each_row() -= arma::mean(Zc, 0); // Centering
  arma::mat S = Zc.t() * Zc;

  // Use safe inverse for initialization
  arma::mat Sigma_init = inv_safe_sympd(S / std::max(1, n) + 1e-2 * arma::eye(p,p));
  arma::mat Omega = inv_safe_sympd(Sigma_init);
  Omega = symmetrize_mat(Omega);

  // --- Hyperparameters & Storage containers ---
  // Use std::vector for list-like structures to avoid Rcpp overhead in loop
  std::vector<arma::vec> theta_list(p);
  std::vector<arma::vec> lambda2_list(p);
  std::vector<arma::vec> nu_list(p);

  arma::vec sigma2 = arma::ones<arma::vec>(p);
  arma::vec tau2 = arma::ones<arma::vec>(p);
  arma::vec xi = arma::ones<arma::vec>(p);

  for (int j = 0; j < p; ++j) {
    int m = p - 1;
    theta_list[j] = arma::zeros<arma::vec>(m);
    lambda2_list[j] = arma::ones<arma::vec>(m);
    nu_list[j] = arma::ones<arma::vec>(m);
  }

  // Storage
  int n_stored = (n_iter - burnin) / thin;
  if ((n_iter - burnin) % thin != 0) n_stored++; // safety if math is slightly off
  if (n_stored < 1) n_stored = 0;

  arma::cube Omegas(p, p, n_stored, arma::fill::zeros);
  arma::cube Zkeep;
  if (store_Z) Zkeep.set_size(n, p, n_stored);

  int store_idx = 0;

  // --- Main Gibbs Loop ---
  for (int it = 0; it < n_iter; ++it) {

    // Step 1: Impute missing values
    arma::mat Sigma = inv_safe_sympd(Omega);
    Z = impute_missing_arma(Y, Z, Sigma);

    // Step 2: Nodewise updates
    // We pass C++ containers directly by reference
    nodewise_horseshoe_update(Z, tau2, lambda2_list, sigma2, nu_list, xi,
                              theta_list, n, p, a0, b0);

    // Step 3: Reconstruct Omega
    Omega.zeros();
    for (int j = 0; j < p; ++j) {
      Omega(j,j) = 1.0 / sigma2(j);
      arma::vec off = - (1.0 / sigma2(j)) * theta_list[j];

      int idx = 0;
      for (int k = 0; k < p; ++k) {
        if (k == j) continue;
        Omega(k, j) = off(idx++);
      }
    }

    // Step 4: Symmetrization
    if (symmetrize == "average") {
      Omega = symmetrize_mat(Omega);
    } else if (symmetrize == "min") {
      // Standard 'AND' rule approximation
      for (int j = 0; j < p; ++j) {
        for (int k = j+1; k < p; ++k) {
          double val = (std::abs(Omega(j,k)) < std::abs(Omega(k,j))) ? Omega(j,k) : Omega(k,j);
          Omega(j,k) = Omega(k,j) = val;
        }
      }
    } else if (symmetrize == "max") {
      // Standard 'OR' rule approximation
      for (int j = 0; j < p; ++j) {
        for (int k = j+1; k < p; ++k) {
          double val = (std::abs(Omega(j,k)) > std::abs(Omega(k,j))) ? Omega(j,k) : Omega(k,j);
          Omega(j,k) = Omega(k,j) = val;
        }
      }
    }

    if (project_spd) {
      arma::mat tmp_chol;
      bool ok = arma::chol(tmp_chol, Omega);
      if (!ok) Omega = nearest_spd(Omega);
    }

    // Storage
    if (it >= burnin && ((it - burnin) % thin == 0)) {
      if (store_idx < n_stored) {
        Omegas.slice(store_idx) = Omega;
        if (store_Z) Zkeep.slice(store_idx) = Z;
        store_idx++;
      }
    }

    // Progress
    if (verbose && ((it+1) % 500 == 0)) {
      Rcout << "Iter " << (it+1) << " / " << n_iter << "\n";
    }
  }

  // Post-processing
  arma::mat Omega_mean = arma::mean(Omegas, 2);
  arma::mat Omega_median = cube_median(Omegas);

  // Return list
  // RcppArmadillo automatically converts arma::cube to R array (3D)
  // No manual flattening needed.
  List out = List::create(
    Named("Omega_draws") = Omegas,
    Named("Omega_mean") = Omega_mean,
    Named("Omega_median") = Omega_median,
    Named("settings") = List::create(Named("n_iter")=n_iter, Named("burnin")=burnin, Named("thin")=thin)
  );

  if (store_Z) {
    out["Z_draws"] = Zkeep;
  }

  return out;
}
