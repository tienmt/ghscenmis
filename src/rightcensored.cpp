// tamtam_fixed_with_right.cpp
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;

// -------- helpers ---------
// safe inverse for symmetric PD matrices: try inv_sympd, fallback to pinv
arma::mat safe_inv_sympd(const arma::mat& A) {
  arma::mat X;
  bool ok = arma::inv_sympd(X, A);
  if (ok) return X;
  // fallback (more stable but slower)
  return arma::pinv(A);
}

// sample from N(mu, sd^2) truncated above at `upper` (i.e. <= upper)
arma::vec rtrunc_norm_upper(int n, const arma::vec& mu, double sd, double upper) {
  arma::vec out(n);
  for (int i = 0; i < n; ++i) {
    double m = mu[i];
    double a = (upper - m) / sd;               // standardized truncation point
    double p_upper = R::pnorm(a, 0.0, 1.0, 1, 0); // CDF at a
    double u = R::runif(0.0, std::max(1e-16, p_upper));
    double z = R::qnorm(u, 0.0, 1.0, 1, 0);
    out[i] = m + sd * z;
  }
  return out;
}

// sample from N(mu, sd^2) truncated below at `lower` (i.e. >= lower)
arma::vec rtrunc_norm_lower(int n, const arma::vec& mu, double sd, double lower) {
  arma::vec out(n);
  for (int i = 0; i < n; ++i) {
    double m = mu[i];
    double a = (lower - m) / sd;               // standardized truncation point
    double p_lower = R::pnorm(a, 0.0, 1.0, 1, 0); // CDF at a
    double low = std::min(std::max(p_lower, 0.0), 1.0 - 1e-16);
    double u = R::runif(low, 1.0);
    double z = R::qnorm(u, 0.0, 1.0, 1, 0);
    out[i] = m + sd * z;
  }
  return out;
}

// SPD check
bool is_spd(const arma::mat& M) {
  arma::mat R;
  return arma::chol(R, M);
}

// Simple nearest SPD via eigenvalue clipping (not full Higham)
arma::mat nearest_spd2(const arma::mat& M) {
  arma::mat symM = 0.5 * (M + M.t());
  arma::vec eigval;
  arma::mat eigvec;
  arma::eig_sym(eigval, eigvec, symM);
  arma::vec eigpos = arma::clamp(eigval, 1e-8, arma::datum::inf);
  return eigvec * arma::diagmat(eigpos) * eigvec.t();
}

// ---------------- nodewise_horseshoe_cpp (adapted from user's original) ----------------

// helper: inverse via safe_inv_sympd

List nodewise_horseshoe_cpp(const arma::mat& Z,
                            arma::vec tau2,
                            List lambda2_list,
                            arma::vec sigma2,
                            List nu_list,
                            arma::vec xi,
                            int n, int p,
                            double a0, double b0) {

  List theta_list(p);

  for (int j = 0; j < p; j++) {
    arma::vec y = Z.col(j);
    arma::mat X = Z;
    X.shed_col(j); // drop column j
    int m = X.n_cols;

    arma::mat XtX = X.t() * X;
    arma::vec Xty = X.t() * y;

    arma::vec lambda2_j = as<arma::vec>(lambda2_list[j]);

    arma::mat A = XtX + arma::diagmat(1.0 / (tau2[j] * lambda2_j));
    arma::mat V = safe_inv_sympd(A);
    arma::vec mu_theta = V * Xty;

    // ensure V is symmetric positive definite for chol
    arma::mat V_stable = 0.5 * (V + V.t());
    if (!is_spd(V_stable)) V_stable += 1e-8 * arma::eye(m, m);

    // Sample theta ~ N(mu_theta, sigma2[j] * V)
    arma::mat cov = sigma2[j] * V_stable;
    arma::mat L = arma::chol(cov, "lower");
    arma::vec z = arma::randn<arma::vec>(m);
    arma::vec theta = mu_theta + L * z;

    theta_list[j] = theta;

    // Update sigma2[j] (inverse-gamma)
    arma::vec res = y - X * theta;
    double rate_sigma = b0 + 0.5 * arma::dot(res, res);
    // R::rgamma(shape, scale)
    double draw = R::rgamma(a0 + 0.5 * n, 1.0 / rate_sigma);
    sigma2[j] = 1.0 / draw;

    // Update lambda2 (local scales)
    arma::vec th2_scaled = arma::square(theta) / (2.0 * sigma2[j] * tau2[j]);
    arma::vec nu_j = as<arma::vec>(nu_list[j]);
    arma::vec rate_lam = 1.0 / nu_j + th2_scaled;
    arma::vec lambda2_new(m);
    for (int k = 0; k < m; ++k) {
      lambda2_new[k] = 1.0 / R::rgamma(1.0, 1.0 / rate_lam[k]);
    }
    lambda2_list[j] = lambda2_new;

    // Update nu
    arma::vec nu_new(m);
    for (int k = 0; k < m; ++k) {
      nu_new[k] = 1.0 / R::rgamma(1.0, 1.0 / (1.0 + 1.0 / lambda2_new[k]));
    }
    nu_list[j] = nu_new;

    // Update tau2[j]
    double sum_term = arma::sum(arma::square(theta) / lambda2_new);
    double rate_tau = (1.0 / xi[j]) + 0.5 * sum_term / sigma2[j];
    tau2[j] = 1.0 / R::rgamma((m + 1.0) / 2.0, 1.0 / rate_tau);

    // Update xi[j]
    xi[j] = 1.0 / R::rgamma(1.0, 1.0 / (1.0 + 1.0 / tau2[j]));
  }

  return List::create(
    _["theta_list"] = theta_list,
    _["tau2"] = tau2,
    _["lambda2_list"] = lambda2_list,
    _["sigma2"] = sigma2,
    _["nu_list"] = nu_list,
    _["xi"] = xi
  );
}

// ---------------- Gibbs sampler for censored graphical horseshoe ----------------
// right_or_left: "left" (observed = max(Z, c)) or "right" (observed = min(Z, c))
// [[Rcpp::export]]
List gibbs_censored_GHS_cpp2(const arma::mat& Ytilde,
                             const arma::vec& cvec,
                             int n_iter = 2000,
                             int burn = 1000,
                             int thin = 1,
                             double a0 = 1e-2,
                             double b0 = 1e-2,
                             std::string symmetrize = "average",
                             bool project_spd = false,
                             bool verbose = true,
                             std::string right_or_left = "left") {

  int n = (int) Ytilde.n_rows;
  int p = (int) Ytilde.n_cols;
  if ((int) cvec.n_elem != p) stop("Length of cvec must equal number of columns");

  // Identify censored entries elementwise depending on left/right censoring
  arma::mat Cmat = arma::repmat(cvec.t(), n, 1);
  arma::umat is_cens;
  arma::umat is_unc;
  if (right_or_left == "left") {
    // observed = max(Z, c): observed == c when Z <= c  --> censored when Ytilde <= c
    is_cens = arma::conv_to<arma::umat>::from(Ytilde <= Cmat);
    is_unc  = 1 - is_cens;
  } else {
    // right censoring: observed = min(Z, c): observed == c when Z >= c  --> censored when Ytilde >= c
    is_cens = arma::conv_to<arma::umat>::from(Ytilde >= Cmat);
    is_unc  = 1 - is_cens;
  }

  // Initialize Z
  arma::mat Z = Ytilde;
  for (int j = 0; j < p; ++j) {
    arma::uvec idx = arma::find(is_cens.col(j));
    if (!idx.is_empty()) {
      for (arma::uword kk = 0; kk < idx.n_elem; ++kk) {
        if (right_or_left == "left") {
          // left-censored: initialize below cutoff
          Z(idx[kk], j) = cvec[j] - std::abs(R::rnorm(0.0, 1.0));
        } else {
          // right-censored: initialize above cutoff
          Z(idx[kk], j) = cvec[j] + std::abs(R::rnorm(0.0, 1.0));
        }
      }
    }
  }

  // Initialize nodewise regression params
  List theta_list(p), lambda2_list(p), nu_list(p);
  arma::vec sigma2(p, arma::fill::ones);
  arma::vec tau2(p, arma::fill::ones);
  arma::vec xi(p, arma::fill::ones);

  for (int j = 0; j < p; ++j) {
    int m = p - 1;
    theta_list[j]   = arma::zeros(m);
    lambda2_list[j] = arma::ones(m);
    nu_list[j]      = arma::ones(m);
  }

  // Storage indices (manual seq with step `thin`)
  std::vector<int> keep_idx;
  for (int it = burn + 1; it <= n_iter; it += thin) keep_idx.push_back(it);
  int n_keep = (int) keep_idx.size();
  arma::cube Omega_draws(p, p, n_keep, arma::fill::zeros);

  int keep_t = 0;

  // Gibbs iterations
  for (int iter = 1; iter <= n_iter; ++iter) {
    // (A) Impute censored Z
    for (int j = 0; j < p; ++j) {
      arma::vec theta_j = as<arma::vec>(theta_list[j]);
      arma::mat X = Z;
      X.shed_col(j);
      arma::vec mu_vec = X * theta_j;
      double sd_j = std::sqrt(sigma2[j]);

      arma::uvec idx_unc = arma::find(is_unc.col(j));
      if (!idx_unc.is_empty()) {
        for (arma::uword kk = 0; kk < idx_unc.n_elem; ++kk) {
          Z(idx_unc[kk], j) = Ytilde(idx_unc[kk], j);
        }
      }

      arma::uvec idx_cen = arma::find(is_cens.col(j));
      if (!idx_cen.is_empty()) {
        arma::vec mu_sel = mu_vec.elem(idx_cen);
        arma::vec z_new;
        if (right_or_left == "left") {
          // left-censored: latent Z <= c_j  -> truncated ABOVE at c_j
          z_new = rtrunc_norm_upper((int) idx_cen.n_elem, mu_sel, sd_j, cvec[j]);
        } else {
          // right-censored: latent Z >= c_j -> truncated BELOW at c_j
          z_new = rtrunc_norm_lower((int) idx_cen.n_elem, mu_sel, sd_j, cvec[j]);
        }
        for (arma::uword kk = 0; kk < idx_cen.n_elem; ++kk) {
          Z(idx_cen[kk], j) = z_new[kk];
        }
      }
    }

    // (B) Node-wise horseshoe regressions
    List res = nodewise_horseshoe_cpp(Z, tau2, lambda2_list, sigma2, nu_list, xi,
                                      n, p, a0, b0);
    theta_list   = res["theta_list"];
    lambda2_list = res["lambda2_list"];
    nu_list      = res["nu_list"];
    tau2   = Rcpp::as<arma::vec>(res["tau2"]);
    sigma2 = Rcpp::as<arma::vec>(res["sigma2"]);
    xi     = Rcpp::as<arma::vec>(res["xi"]);


    // (C) Reconstruct Omega
    arma::mat Omega(p, p, arma::fill::zeros);
    for (int j = 0; j < p; ++j) {
      Omega(j, j) = 1.0 / sigma2[j];
      arma::vec theta_j = as<arma::vec>(theta_list[j]);
      arma::vec off = -(1.0 / sigma2[j]) * theta_j;
      arma::uword idx = 0;
      for (int k = 0; k < p; ++k) {
        if (k == j) continue;
        Omega(k, j) = off[idx++];
      }
    }

    // Symmetrize
    if (symmetrize == "min" || symmetrize == "max") {
      for (int j = 0; j < p; ++j) {
        for (int k = j + 1; k < p; ++k) {
          double val = (symmetrize == "min") ? std::min(Omega(j, k), Omega(k, j)) : std::max(Omega(j, k), Omega(k, j));
          Omega(j, k) = Omega(k, j) = val;
        }
      }
    } else {
      Omega = 0.5 * (Omega + Omega.t());
    }

    if (project_spd && !is_spd(Omega)) {
      Omega = nearest_spd2(Omega);
    }

    // Store
    if (std::find(keep_idx.begin(), keep_idx.end(), iter) != keep_idx.end()) {
      if (keep_t < n_keep) {
        Omega_draws.slice(keep_t) = Omega;
        ++keep_t;
      }
    }

    if (verbose && (iter % std::max(1, n_iter / 10) == 0)) {
      Rcpp::Rcout << "Iter " << iter << " / " << n_iter << "\n";
    }
  }

  arma::mat Omega_post_mean = arma::zeros(p, p);
  if (n_keep > 0) Omega_post_mean = arma::mean(Omega_draws, 2);

  return List::create(
    _["Omega_draws"] = Omega_draws,
    _["Omega_mean"]  = Omega_post_mean,
    _["Z_last"]      = Z,
    _["theta_last"]  = theta_list,
    _["sigma2_last"] = sigma2,
    _["lambda2_last"] = lambda2_list,
    _["tau2"]   = tau2,
    _["settings"]    = List::create(_["n_iter"]=n_iter, _["burn"]=burn, _["thin"]=thin,
                            _["a0"]=a0, _["b0"]=b0,
                            _["symmetrize"]=symmetrize,
                            _["project_spd"]=project_spd,
                            _["right_or_left"]=right_or_left)
  );
}
