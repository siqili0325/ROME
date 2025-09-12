library(MASS)
library(tidyverse)
library(stats)

g.logit <- function(x) exp(x)/(1 + exp(x))
softmax <- function(eta) exp(eta) / sum(exp(eta))

simulateDataSoftmax <- function(n, p, p_S, G,index_OS,
                                gamma,
                                beta,
                                Sig.X,
                                misspecify=F,
                                rand_fake=0.1,
                                family = rep("gaussian", p),
                                ones_rate = rep(NA, p),
                                myseed = 123,
                                lm=T) {
  set.seed(myseed)
  X_raw <- mvrnorm(n, mu = rep(0, p), Sigma = Sig.X)
  X_df <- as.data.frame(X_raw)
  for (i in seq_len(p)) {
    if (family[i] == "binomial") {
      cutoff <- qnorm(1 - (ones_rate[i] / 2))
      X_df[[i]] <- ifelse(X_df[[i]] > cutoff, 1, 0)
    }
  }
  colnames(X_df) <- c(paste0("A", 1:(p - p_S)), paste0("S", 1:p_S))
  
  A_df <- X_df[, 1:(p - p_S), drop = FALSE]
  S_df <- X_df[, (p - p_S + 1):p, drop = FALSE]
  S_mat <- as.matrix(S_df) # here still using all S
  group_label <- integer(n)
  for (i in seq_len(n)) {
    eta <- numeric(G)
    for (j in 1:G) {
      if (misspecify) {
        eta[j] <- (gamma[j, 1] * sin(S_mat[i, 1])) +  
          (gamma[j, 2] * S_mat[i, 2]) +
          (gamma[j, 3] * (S_mat[i, 3] * S_mat[i, 4]))
      } else {
        eta[j] <- sum(gamma[j, ] * S_mat[i, ])
      }
    }
    probs <- softmax(eta)
    group_label[i] <- sample.int(G, size = 1, prob = probs)
  }

  X_df2=X_df[, c(1:p_A,p_A+index_OS), drop = FALSE]
  Y_vec <- numeric(n)
  for (i in seq_len(n)) {
    g_idx <- group_label[i]
    X_i <- c(1, as.numeric(X_df2[i, ]))
    linpred <- sum(beta[g_idx, ] * X_i)
    if(lm){
      Y_vec[i] <- linpred + rnorm(1, mean = 0, sd = 0.1)
    }
    else{
      p_y1 <- g.logit(linpred)
      Y_vec[i] <- rbinom(1, size = 1, prob = p_y1)
    }
  }
  
  G0_fake=get.fake(group_label, G, rand_fake)
  
  dat <- data.frame(Y = Y_vec, X_df, G0 = group_label,G0_fake=G0_fake)
  return(list(dat = dat, gamma = gamma, beta = beta, group_label = group_label))
}


get.fake <- function(vec, G, rand_fake) {
  fake_vec <- vec
  flip <- rbinom(length(vec), size = 1, prob = rand_fake) == 1
  for (i in which(flip)) {
    alternatives <- setdiff(1:G, vec[i])
    fake_vec[i] <- sample(alternatives, size = 1)
  }
  return(fake_vec)
}

epsilon.gen <- function(delta, G, p, uni_group) {
  # For example, narrower range in [1 - delta, 1 + delta]
  # If delta=0.2, then range is [0.8, 1.2]
  if (uni_group) {
    fac <- runif(G, min = 1 - delta, max = 1 + delta)
    mat <- matrix(rep(fac, p), nrow = G, ncol = p, byrow = FALSE)
  } else {
    # mat <- matrix(runif(G * p, min = 1 - delta, max = 1 + delta), nrow = G, ncol = p)
    mat <- matrix(runif(G * p, min = 1 - delta, max = 1 + delta), nrow = G, ncol = p) *
      matrix(sample(c(-1, 1), G * p, replace = TRUE), nrow = G, ncol = p)
  }
  return(mat)
}


