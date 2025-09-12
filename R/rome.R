library(CVXR)

EM.iter.lm <- function(Y, A, S, index_OS,index_OS2,initial_labels = NULL, G, 
                       max_iter = 100, tol1 = 1e-6, tol2 = 5e-3, 
                       alpha = 0.5, myseed = 123) {
  set.seed(myseed)
  A <- as.data.frame(A)
  if (ncol(A) > 0) {
    colnames(A) <- paste0("A", seq_len(ncol(A)))
  }
  S <- as.data.frame(S)
  if (ncol(S) > 0) {
    colnames(S) <- paste0("S", seq_len(ncol(S)))
  }
  
  n <- length(Y)    
  pA <- ncol(A) 
  # pS <- ncol(S)
  pX <- 1 + pA + length(index_OS) 
  
  if (is.null(initial_labels)) {
    initial_labels <- sample(1:G, n, replace = TRUE)
  }
  
  X_membership <- as.matrix(S)
  S_sub <- S[, index_OS, drop = FALSE]
  X_outcome <- as.matrix(cbind(1, A, S_sub))
  
  S2=S[, index_OS2, drop = FALSE]
  pS=ncol(S2)
  gamma <- matrix(0, nrow = G, ncol = pS)
  for (j in 1:G) {
    group_indicator <- as.integer(initial_labels == j)
    fit_gamma <- glm(group_indicator ~ . - 1, data = S2, family = binomial(link = "logit"))
    gamma[j, ] <- coef(fit_gamma)
  }
  
  omega <- matrix(0, nrow = G, ncol = pX)
  for (j in 1:G) {
    idx <- which(initial_labels == j)
    if (length(idx) > 5) {
      data_group <- data.frame(Y = Y[idx],
                               A[idx, , drop = FALSE],
                               S_sub[idx, , drop = FALSE])
      fit_beta <- lm(Y ~ ., data = data_group)  
      omega[j, ] <- coef(fit_beta)
    }
  }
  overall_fit <- lm(Y ~ ., data = data.frame(Y, A, S_sub))
  overall_beta <- coef(overall_fit)
  for (j in 1:G) {
    if (any(is.na(omega[j, ]))) {
      omega[j, ] <- overall_beta
    }
  }
  
  theta <- matrix(1 / G, nrow = n, ncol = G)

  IRLS_logistic <- function(X, y, w, beta_init, max_irls = 25, irls_tol = 1e-5) {
    n <- nrow(X)
    d <- ncol(X)
    beta <- beta_init
    clamp <- function(eta, limit = 30) pmax(pmin(eta, limit), -limit)
    logistic <- function(z) 1 / (1 + exp(-z))
    ridge <- 1e-6  
    for (it in seq_len(max_irls)) {
      eta <- as.vector(X %*% beta)
      eta <- clamp(eta, 30)
      p <- logistic(eta)
      grad <- t(X) %*% (w * (y - p))
      W_vec <- as.numeric(w * p * (1 - p))
      Xw <- sweep(X, 1, W_vec, `*`)
      H <- - t(X) %*% Xw
      H_reg <- H + diag(ridge, d)
      delta <- tryCatch(
        solve(-H_reg, grad),
        error = function(e) NA
      )
      if (any(is.na(delta))) {
        ridge <- ridge * 10
        if (ridge > 1e3) {
          message("IRLS: Hessian singular; returning current beta.")
          return(beta)
        }
        next
      }
      beta_new <- beta + delta
      diff_val <- sum(abs(beta_new - beta))
      beta <- beta_new
      if (diff_val < irls_tol) break
    }
    beta
  }
  
  E_step <- function() {
    for (i in seq_len(n)) {
      S_i <- as.numeric(S2[i, ])
      S_i2 <- as.numeric(S_sub[i, ])
      A_i <- if (pA > 0) as.numeric(A[i, ]) else numeric(0)
      # Build design vector: [1, A, S]
      X_i <- c(1, A_i, S_i2)
      numer <- numeric(G)
      for (j in seq_len(G)) {
        mem_j <- exp(sum(gamma[j, ] * S_i))
        L_out <- exp(-0.5 * (Y[i] - sum(omega[j, ] * X_i))^2)
        numer[j] <- mem_j * L_out
      }
      theta[i, ] <<- numer / (sum(numer) + 1e-15)
    }
  }
  
  compute_ll <- function() {
    ll <- 0
    for (i in seq_len(n)) {
      S_i <- as.numeric(S2[i, ])
      S_i2 <- as.numeric(S_sub[i, ])
      A_i <- if (pA > 0) as.numeric(A[i, ]) else numeric(0)
      X_i <- c(1, A_i, S_i2)
      temp <- 0
      for (j in seq_len(G)) {
        mem_j <- exp(sum(gamma[j, ] * S_i))
        L_out <- exp(-0.5 * (Y[i] - sum(omega[j, ] * X_i))^2)
        temp <- temp + mem_j * L_out
      }
      ll <- ll + log(temp + 1e-15)
    }
    ll
  }
  
  M_step <- function() {
    old_gamma <- gamma
    old_omega <- omega
    old_ll <- compute_ll()

    for (j in seq_len(G)) {
      df_gamma <- data.frame(S2)
      y_gamma <- theta[, j]
      fit_gamma <- glm(cbind(success = y_gamma, failure = 1 - y_gamma) ~ . - 1, 
                       data = df_gamma, 
                       family = quasibinomial(link = "logit"))
      new_gamma_j <- coef(fit_gamma)
      alpha_gamma <- alpha
      candidate_gamma_j <- old_gamma[j, ] + alpha_gamma * (new_gamma_j - old_gamma[j, ])
      gamma[j, ] <<- candidate_gamma_j
      candidate_ll <- compute_ll()
      while (candidate_ll < old_ll && alpha_gamma > tol2) {
        alpha_gamma <- alpha_gamma / 2
        candidate_gamma_j <- old_gamma[j, ] + alpha_gamma * (new_gamma_j - old_gamma[j, ])
        gamma[j, ] <<- candidate_gamma_j
        candidate_ll <- compute_ll()
      }
      gamma[j, ] <<- candidate_gamma_j
    }

    for (j in seq_len(G)) {
      w_j <- theta[, j]
      Xw <- sweep(X_outcome, 1, w_j, `*`)
      XtX <- t(X_outcome) %*% Xw
      XtY <- t(X_outcome) %*% (w_j * Y)
      new_omega_j <- tryCatch(solve(XtX, XtY), error = function(e) old_omega[j, ])
      
      alpha_omega <- alpha
      candidate_omega_j <- old_omega[j, ] + alpha_omega * (new_omega_j - old_omega[j, ])
      omega[j, ] <<- candidate_omega_j
      candidate_ll <- compute_ll()
      while (candidate_ll < old_ll && alpha_omega > tol2) {
        alpha_omega <- alpha_omega / 2
        candidate_omega_j <- old_omega[j, ] + alpha_omega * (new_omega_j - old_omega[j, ])
        omega[j, ] <<- candidate_omega_j
        candidate_ll <- compute_ll()
      }
      omega[j, ] <<- candidate_omega_j
    }
  }
  
  for (iter in seq_len(max_iter)) {
    cat("Iteration:", iter, "\n")
    E_step()
    old_gamma <- gamma
    old_omega <- omega
    M_step()
    param_diff <- sum(abs(gamma - old_gamma)) + sum(abs(omega - old_omega))
    cat("Parameter change:", param_diff, "\n")
    if (iter > 1 && param_diff < tol1) {
      cat("Convergence reached after", iter, "iterations.\n")
      break
    }
  }
  
  final_ll <- compute_ll()
  num_params <- G * pS + G * pX
  AIC_val <- -2 * final_ll + 2 * num_params
  BIC_val <- -2 * final_ll + num_params * log(n)
  
  return(list(
    gamma = gamma,
    omega = omega,
    theta = theta,
    logLik = final_ll,
    AIC = AIC_val,
    BIC = BIC_val
  ))
}


# compute_group_predictions <- function(test_A, test_S, omega) {
#   # test_A: test data matrix for A (n_test x pA)
#   # test_S: test data matrix for S (n_test x pS)
#   # omega: outcome parameters, G x (pA + pS)
#   
#   n_test <- nrow(test_A)
#   G <- nrow(omega)
#   pA <- ncol(test_A)
#   pS <- ncol(test_S)
#   
#   X_test <- cbind(test_A, test_S)  # n_test x (pA+pS)
#   logistic <- function(x) 1 / (1 + exp(-x))
#   
#   pred.mat <- matrix(NA, nrow = n_test, ncol = G)
#   for (j in 1:G) {
#     linpred <- X_test %*% omega[j, ]
#     pred.mat[, j] <- logistic(linpred)
#   }
#   return(pred.mat)
# }

bias_correct <- function(fk, fl, wl, Xl, Yl){
  nl = nrow(Xl)
  fkX = as.matrix(cbind(1,Xl)) %*% fk
  flX = as.matrix(cbind(1,Xl)) %*% fl
  return(mean(wl*fkX*(flX - Yl)))
}

compute_v <- function(Gamma, L, const, v0){
  v = Variable(L)
  obj = quad_form(v, Gamma)
  constraints = list(v>=0, sum(v)==1, cvxr_norm(v-v0)/sqrt(L)<= const)
  prob = Problem(Minimize(obj), constraints)
  result = solve(prob)
  v.opt = result$getValue(v)
  reward = result$value
  return(list(v = v.opt,
              reward = reward))
}

compute_maximin.2 <- function(X, Y, pred.mat, omega, n0, G0, v0, consts.set = c(1, 0.4, seq(0.3, 0, by = -0.01))) {
  tGamma <- t(pred.mat) %*% pred.mat / n0
  L <- ncol(pred.mat)
  hGamma <- tGamma
  
  for (k in 1:L) {
    for (l in 1:L) {
      map.k <- (G0 == k)
      map.l <- (G0 == l)
      
      if (sum(map.k) > 0 && sum(map.l) > 0) {
        f.k <- omega[k, ]
        f.l <- omega[l, ]
        
        num1 <- bias_correct(f.k, f.l, rep(1, sum(map.l)), X[map.l, ], Y[map.l])
        num2 <- bias_correct(f.l, f.k, rep(1, sum(map.k)), X[map.k, ], Y[map.k])
        if (is.finite(num1) && is.finite(num2)) {
          hGamma[k, l] <- hGamma[k, l] - (num1 + num2)
        }
      }
    }
  }
  
  eig_decomp <- eigen(hGamma)
  eig_val <- eig_decomp$values
  eig_vec <- eig_decomp$vectors
  eig_val_real <- Re(eig_val)
  eig_val_clean <- pmax(eig_val_real, 1e-6)
  hGamma_clean <- Re(eig_vec %*% diag(eig_val_clean) %*% Conj(t(eig_vec)))
  
  print("Final Cleaned hGamma Matrix:")
  print(hGamma_clean)
  
  return(hGamma_clean)
}




