

#########################
### construct data
###########################
rm(list=ls())

df <- read.table('./data3/HS_FSL.txt', header = T)
# df <- read.table('./data/bart_exampleData.txt', header = T)
subjs <- unique(df$subjID)
n_subj <- length(subjs)
t_max <- max(df$trial)
t_subjs <- array(0, n_subj)
pumps     <- array(0, c(n_subj, t_max))
explosion <- array(0, c(n_subj, t_max))
L <- array(0, c(n_subj, t_max))

# Write from df to the data arrays
for (i in 1:n_subj) {
  subj <- subjs[i]
  DT_subj <- subset(df, subjID == subj)
  t_subjs[i] <- length(DT_subj$trial)
  t <- t_subjs[i]
  pumps[i, 1:t]     <- DT_subj$pumps
  explosion[i, 1:t] <- DT_subj$explosion
  L[i, 1:t] <- pumps[i, 1:t] + 1 - explosion[i, 1:t]
}

L[L==12] <- 11

# r=0:max(pumps)
r_accu = c(0.00, 0.05, 0.15, 0.25, 0.55, 0.95, 1.45, 2.05, 2.75, 3.45, 4.25, 5.15)
r      = c()
for (j in 1:max(L)) {
  r[j] <- r_accu[j+1]-r_accu[j]
}

# Wrap into a list for Stan
dataList <- list(
  N         = n_subj,
  T         = t_max,
  Tsubj     = t_subjs,
  P         = max(pumps) + 1,
  pumps     = pumps,
  explosion = explosion,
  r         = r,
  r_accu    = r_accu,
  L         = L
)

###############################
### fit model
###############################
library(rstan)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
# options(mc.cores = 4)

nIter     = 2000   # 2000 for test, 4000 for real data
nChains   = 4 
nWarmup   = floor(nIter/2)
nThin     = 1

###------------------------------------
### model 1
modelFile1 = './scripts/bart_par5_5.0.stan'

cat("Estimating", modelFile1, "model... \n")
startTime = Sys.time(); print(startTime)
cat("Calling", nChains, "simulations in Stan... \n")

fit_1 = stan(
  modelFile1,
  data    = dataList, 
  chains  = nChains,
  iter    = nIter,
  warmup  = nWarmup,
  thin    = nThin,
  init    = "random",
  # control = list(adapt_delta = 0.999, max_treedepth = 20),
  # control  = list(adapt_delta=0.999, max_treedepth=100),
  seed    = 233
)

cat("Finishing", modelFile1, "model simulation ... \n")
endTime = Sys.time(); print(endTime)  
cat("It took",as.character.Date(endTime - startTime), "\n")

###------------------------------------
### model 2 
modelFile2 = './scripts/bart_par5_MV_WYM.stan'

cat("Estimating", modelFile2, "model... \n")
startTime = Sys.time(); print(startTime)
cat("Calling", nChains, "simulations in Stan... \n")

fit_2 = stan(
  modelFile2,
  data    = dataList, 
  chains  = nChains,
  iter    = nIter,
  warmup  = nWarmup,
  thin    = nThin,
  init    = "random",
  # control = list(adapt_delta = 0.999, max_treedepth = 20),
  # control  = list(adapt_delta=0.999, max_treedepth=100),
  seed    = 123456
)

cat("Finishing", modelFile2, "model simulation ... \n")
endTime = Sys.time(); print(endTime)  
cat("It took",as.character.Date(endTime - startTime), "\n")

#########################
### model diagnosis
##########################

library(loo)
fit_1_u <- fit_1
fit_1_m <- fit_1
LL_1 = extract_log_lik(fit_1_u)
LL_2 = extract_log_lik(fit_1_m)

# loo(LL_1)
loo_1 = loo(LL_1)
loo_2 = loo(LL_2)

loo_compare(loo_1, loo_2) # positive difference indicates the 2nd model's predictive accuracy is higher


traceplot(fit_1, pars = c('mu_psi', 'mu_xi', 'mu_rho', 'mu_tau', 'mu_lambda'), inc_warmup = F, nrow = 2)
traceplot(fit_2, pars = c('mu_phi', 'mu_eta', 'mu_gam', 'mu_tau'), inc_warmup = F, nrow = 2)
# stan_plot(fit_hc_con4, pars = c('alpha', 'beta'), show_density = T)
stan_plot(fit_1, pars = c('mu_psi', 'mu_xi', 'mu_rho', 'mu_tau', 'mu_lambda'), show_density = T)
stan_plot(fit_2, pars = c('mu_phi', 'mu_eta', 'mu_gam', 'mu_tau'), show_density = T)

#########################
### group compare
##########################
# load results
load(file = 'results/MIXMDD0104_par5_vs_par4.Rdata')
fit_1_mdd <- fit_1
fit_2_mdd <- fit_2
load(file = 'results/hc_66_par5_vs_par4_v34.Rdata')

fit_1_hc <- fit_1
fit_2_hc <- fit_2

# plot by group
stan_plot(fit_1_mdd, pars = c('mu_psi', 'mu_xi', 'mu_rho', 'mu_lambda'), show_density = T)
stan_plot(fit_1_hc, pars = c('mu_psi', 'mu_xi', 'mu_rho', 'mu_lambda'), show_density = T)

stan_plot(fit_2_mdd, pars = c('mu_phi', 'mu_eta', 'mu_gam'), show_density = T)
stan_plot(fit_2_hc, pars = c('mu_phi', 'mu_eta', 'mu_gam'), show_density = T)

# extract parameter

rho_hc <- extract(fit_1_hc)$mu_rho
rho_mdd <- extract(fit_1_mdd)$mu_rho

lambda_hc <- extract(fit_1_hc)$mu_lambda
lambda_mdd <- extract(fit_1_mdd)$mu_lambda

psi_mdd <- extract(fit_1_mdd)$mu_psi
psi_hc <- extract(fit_1_hc)$mu_psi

xi_hc <- extract(fit_1_hc)$mu_xi
xi_mdd <- extract(fit_1_mdd)$mu_xi

tau_hc <- extract(fit_1_hc)$mu_tau
tau_mdd <- extract(fit_1_mdd)$mu_tau

phi_hc <- extract(fit_2_hc)$mu_phi
phi_mdd <- extract(fit_2_mdd)$mu_phi

eta_hc <- extract(fit_2_hc)$mu_eta
eta_mdd <- extract(fit_2_mdd)$mu_eta

gam_hc <- extract(fit_2_hc)$mu_gam
gam_mdd <- extract(fit_2_mdd)$mu_gam

tau_hc <- extract(fit_2_hc)$mu_tau
tau_mdd <- extract(fit_2_mdd)$mu_tau

# compute 95% CI
t.test(rho_mdd, rho_hc)
t.test(lambda_mdd, lambda_hc)
t.test(psi_mdd, psi_hc)
t.test(xi_mdd, xi_hc)
t.test(tau_mdd, tau_hc)
sd(rho_mdd)
sd(rho_hc)
sd(lambda_mdd)
sd(lambda_hc)
sd(psi_mdd)
sd(psi_hc)
sd(xi_mdd)
sd(xi_hc)
sd(tau_mdd)
sd(tau_hc)

t.test(phi_mdd, phi_hc)
t.test(eta_mdd, eta_hc)
t.test(gam_mdd, gam_hc)
t.test(tau_mdd, tau_hc)

# plot 95% CI of difference
library(hBayesDM)
mean(rho_mdd - rho_hc)
mean(lambda_mdd - lambda_hc)
mean(psi_mdd - psi_hc)
mean(xi_mdd - xi_hc)
mean(tau_mdd - tau_hc)

mean(phi_mdd - phi_hc)
mean(tau_mdd - tau_hc)
mean(eta_mdd - eta_hc)
mean(gam_mdd - gam_hc)


plotHDI(rho_mdd - rho_hc) + geom_density(color = "red", linetype = 4, adjust = 1.75, lwd = 1) + 
  scale_x_continuous(limits = c(0, 0.2))
plotHDI(lambda_mdd - lambda_hc)+ geom_density(color = "red", linetype = 4, adjust = 1.75, lwd = 1) 
plotHDI(psi_mdd- psi_hc)
plotHDI(xi_mdd - xi_hc)+ geom_density(color = "red", linetype = 4, adjust = 1.75, lwd = 1) 
plotHDI(tau_mdd - tau_hc)

plotHDI(phi_mdd - phi_hc)
plotHDI(tau_mdd - tau_hc)
plotHDI(eta_mdd- eta_hc)
plotHDI(gam_mdd - gam_hc)

rho_hc <- c(colMeans(extract(fit_1_hc,pars='rho')$rho))
write.csv(rho_hc,file="rho_fsl.csv")
lambda_hc <- c(colMeans(extract(fit_1_hc,pars='lambda')$lambda))
write.csv(lambda_hc,file="lambda_fsl.csv")
psi_hc <- c(colMeans(extract(fit_1_hc,pars='psi')$psi))
write.csv(psi_hc,file="psi_fsl.csv")
xi_hc <- c(colMeans(extract(fit_1_hc,pars='xi')$xi))
write.csv(xi_hc,file="xi_fsl.csv")
tau_hc <- c(colMeans(extract(fit_1_hc,pars='tau')$tau))
write.csv(tau_hc,file="tau_fsl.csv")
######################## posterior predictive check#####################
library(rstan)

load(file = 'hc_66_par5_vs_par4_v34.Rdata')
load(file = 'results/MIXMDD0104_par5_vs_par4.Rdata')

# average MCMC samples
y_pred_1 <- extract(fit_1)$y_pred
y_pred_2 <- extract(fit_2)$y_pred
dims <- dim(y_pred_1)
y_pred_avg_1 <- array(dim = c(dims[2:4]))
y_pred_avg_2 <- array(dim = c(dims[2:4]))

for (i in 1:dims[2]) {
  for (j in 1:dims[3]) {
    for (k in 1:dims[4]) {
      y_pred_avg_1[i,j,k] <- sum(y_pred_1[,i,j,k])/dims[1]
      y_pred_avg_2[i,j,k] <- sum(y_pred_2[,i,j,k])/dims[1]
    }
  }
}

# real data
df <- read.table('./data/BART_BayesModel_hc_66.txt', header = T)
df <- read.table('./data/MIXMDD0104.txt', header = T)
subjs <- unique(df$subjID)
n_subj <- length(subjs)
t_max <- max(df$trial)
t_subjs <- array(0, n_subj)
pumps     <- array(0, c(n_subj, t_max))

for (i in 1:n_subj) {
  subj <- subjs[i]
  DT_subj <- subset(df, subjID == subj)
  t_subjs[i] <- length(DT_subj$trial)
  t <- t_subjs[i]
  pumps[i, 1:t]     <- DT_subj$pumps
}

y_real <- array(-1, c(n_subj, t_max, 12))

for (j in 1:n_subj) {
  for (k in 1:t_subjs[j]) {
    for (l in 1:12) {
      if(l <= pumps[j,k])
        y_real[j,k,l] = 1 else
          y_real[j,k,l] = 0
    }
  }
}

y_real[y_pred_avg_1 == -1] = -1

# average probility of pump by decision times
y_pred_avg_l_1 <- c()
y_pred_avg_l_2 <- c()
y_real_avg_l <- c()

for (l in 1:dims[4]) {
  y_pred_avg_l_1[l] <- mean(y_pred_avg_1[,,l][y_pred_avg_1[,,l]!=-1])
  y_pred_avg_l_2[l] <- mean(y_pred_avg_2[,,l][y_pred_avg_2[,,l]!=-1])
  y_real_avg_l[l]   <- mean(y_real[,,l][y_real[,,l]!=-1])
}

plot(as.vector(y_real_avg_l), type='o', ylab='probability of pump', main='average probability of pump as balloon bigger') +
lines(y_pred_avg_l_1, type = "o", col = "blue")# +
lines(y_pred_avg_l_2, type = "o", col = "red")

# average probility of pump by trails
# y_pred_avg_pumps_1 <- c()
# y_pred_avg_pumps_2 <- c()
# y_real_avg_pumps <- c()
# 
# for (m in 1:dims[3]) {
#   y_pred_avg_pumps_1[m] <- mean(y_pred_avg_1[,m,][y_pred_avg_1[,m,]!=-1])
#   y_pred_avg_pumps_2[m] <- mean(y_pred_avg_2[,m,][y_pred_avg_2[,m,]!=-1])
#   y_real_avg_pumps[m]   <- mean(y_real[,m,][y_real[,m,]!=-1])
# }

# correlation between real choice and predictive choice
y_real_flat <- c(y_real[y_real != -1])
y_pred_avg_flat_1 <- c(y_pred_avg_1[y_pred_avg_1 != -1])
y_pred_avg_flat_2 <- c(y_pred_avg_2[y_pred_avg_2 != -1])

cor.test(y_real_flat, y_pred_avg_flat_1)
cor.test(y_real_flat, y_pred_avg_flat_2)

