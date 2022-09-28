
data {
  int<lower=1> N;            // Number of subjects
  int<lower=1> T;            // Maximum number of trials
  int<lower=0> Tsubj[N];     // Number of trials for each subject
  int<lower=2> P;            // Number of max pump + 1 ** CAUTION **
  int<lower=0> L[N, T];      // Number of decision times
  int<lower=0> pumps[N, T];  // Number of pump
  int<lower=0,upper=1> explosion[N, T];  // Whether the balloon exploded (0 or 1)
  real r[P-1];
  real r_accu[P];
  
}

transformed data{
  // Whether a subject pump the button or not (0 or 1) participants/j  trial/k  pumps+1/P  current pumps/l
  int d[N, T, P];

  for (j in 1:N) {
    for (k in 1:Tsubj[j]) {
      for (l in 1:P) {
        if (l <= pumps[j, k])
          d[j, k, l] = 1;
        else
          d[j, k, l] = 0;
      }
    }
  }
}

parameters {
  // Group-level parameters
  vector[5] mu_pr;
  vector<lower=0>[5] sigma;

  // Normally distributed error for Matt trick 
  vector[N] psi_pr;
  vector[N] xi_pr;
  vector[N] rho_pr;
  vector[N] tau_pr;
  vector[N] lambda_pr;
}

transformed parameters {
  // Subject-level parameters with Matt trick
  vector<lower=0,upper=1>[N] psi;   // initial belief of burst
  vector<lower=0>[N] xi;   // learning rate
  vector<lower=0,upper=2>[N] rho;   // risk taking
  vector<lower=0>[N] tau;   // inverse temp
  vector<lower=0>[N] lambda;   // loss aversion

  psi = Phi_approx(mu_pr[1] + sigma[1] * psi_pr);
  xi = exp(mu_pr[2] + sigma[2] * xi_pr);
  rho = Phi_approx(mu_pr[3] + sigma[3] * rho_pr)*2;
  tau = exp(mu_pr[4] + sigma[4] * tau_pr);
  lambda = exp(mu_pr[5] + sigma[5] * lambda_pr);
}

model {
  // Prior
  mu_pr  ~ normal(0, 1);
  sigma ~ normal(0, 0.2);

  psi_pr ~ normal(0, 1);
  xi_pr ~ normal(0, 1);
  rho_pr ~ normal(0, 1);
  tau_pr ~ normal(0, 1);
  lambda_pr ~ normal(0, 1);

  // Likelihood
  for (j in 1:N) {
    // Initialize n_succ and n_pump and previous probability of pump for a subject
    real n_succ = 0;  // Number of successful pumps
    real n_pump = 0;  // Number of total pumps
    real Pre_prob_pump = 0; //previous probability of pumps
    // int r = 1;   // the amount of reward for each sucessful pump
    

    for (k in 1:Tsubj[j]) {
      real p_burst;  // Belief on a balloon to be burst
      real U_pump;   // subjective utilities for pumping and not pumping
         
      p_burst = exp(-xi[j] * n_pump) * psi[j] + (1 - exp(-xi[j] * n_pump)) * Pre_prob_pump;
     
      // Calculate likelihood with bernoulli distribution
      for (l in 1:L[j,k]){
        U_pump = (1 - p_burst) * (r[l] ^ rho[j]) - p_burst * lambda[j] * (r_accu[l] ^ rho[j]);
        d[j, k, l] ~ bernoulli_logit(tau[j] * U_pump);
      }
  
      // Update n_succ and n_pump and probability of pumps after each trial ends
      n_succ += pumps[j, k] - explosion[j, k];
      n_pump += pumps[j, k];
      if (n_pump>0){
          Pre_prob_pump = (n_pump - n_succ) /n_pump;
      }
    }
  }
}

generated quantities {
  // Actual group-level mean
  real<lower=0, upper=1> mu_psi = Phi_approx(mu_pr[1]);
  real<lower=0, upper=2> mu_rho = Phi_approx(mu_pr[3])*2;
  real<lower=0> mu_xi = exp(mu_pr[2]);
  real<lower=0> mu_tau = exp(mu_pr[4]);
  real<lower=0> mu_lambda = exp(mu_pr[5]);

  // Log-likelihood for model fit
  real log_lik[N];

  // For posterior predictive check
  real y_pred[N, T, P];

  // Set all posterior predictions to 0 (avoids NULL values)
  for (j in 1:N)
    for (k in 1:T)
      for(l in 1:P)
        y_pred[j, k, l] = -1;

  { // Local section to save time and space
    for (j in 1:N) {
      // Initialize n_succ and n_pump and previous probability of pump for a subject
      real n_succ = 0;  // Number of successful pumps
      real n_pump = 0;  // Number of total pumps
      real Pre_prob_pump = 0; //previous probability of pumps
      // int r = 1;   // the amount of reward for each sucessful pump

      log_lik[j] = 0;
  
      for (k in 1:Tsubj[j]) {
        real p_burst;  // Belief on a balloon to be burst
        real U_pump;   // subjective utilities for pumping and not pumping
           
        p_burst = exp(-xi[j] * n_pump) * psi[j] + (1 - exp(-xi[j] * n_pump)) * Pre_prob_pump;
       
        // Calculate likelihood with bernoulli distribution
        for (l in 1:L[j,k]){
           U_pump = (1 - p_burst) * (r[l] ^ rho[j]) - p_burst * lambda[j] * (r_accu[l] ^ rho[j]);
          log_lik[j] += bernoulli_logit_lpmf(d[j, k, l] | tau[j] * U_pump);
          y_pred[j, k, l] = bernoulli_logit_rng(tau[j] * U_pump);
        }
    
        // Update n_succ and n_pump and probability of pumps after each trial ends
        n_succ += pumps[j, k] - explosion[j, k];
        n_pump += pumps[j, k];
        Pre_prob_pump = (n_pump - n_succ) /n_pump;
      }
    }
  }
}

