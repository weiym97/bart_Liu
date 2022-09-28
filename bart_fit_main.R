###s######################
### construct data
###########################
rm(list=ls())
args <- commandArgs(trailingOnly = TRUE)
#args <- c('simulation','EWBart','EWBart_simulation_Liu')

preprocessing <- function(data){
 # Only preserve trials with pump number no less than 2
  data <- data[data$pumps>=1,]
  return(data)
}

extract_posterior <- function(subjs,result){
  n_subj <- length(subjs)
  param_name <- unique(gsub("\\[.*","",x=row.names(result)))
  posterior <-data.frame(matrix(ncol=length(param_name)+1,nrow=n_subj, dimnames=list(NULL, c('subjID',param_name))))
  for (i in 1:n_subj) {
    posterior[i,'subjID'] <- subjs[i]
    for (j in 1:length(param_name)){
      temp_name <- paste(param_name[j],'[',i,']',sep='')
      posterior[i,param_name[j]] = result[temp_name,'mean']
    }
  }
  return(posterior)
}

return_param <- function(model_name){
  params=switch(EXPR=model_name,
                FourparamBart=c('phi','eta','gamma','tau'),
                EWBart = c('psi','xi','rho','lambda','tau'),
                EWMVBart = c('psi','xi','rho','lambda','tau'),
                PTBart_9=c('psi','xi','gamma','tau','lambda'),
                PTBart_10=c('psi','xi','gamma','tau','lambda'),
                PTBart_20=c('psi','xi','lambda','tau'),
                PTBart_100=c('psi','xi','gamma','tau','lambda'),
                PTBart_101=c('psi','xi','gamma','tau','lambda','alpha'),
                PTBart_102=c('psi','xi','gamma','tau','lambda','alpha','beta'),
                PTBart_103=c('psi','xi','gamma','tau','lambda','alpha'),
                PTBart_104=c('psi','xi','gamma','tau','lambda','alpha'),
                PTBart_105=c('psi','xi','gamma','tau','lambda','alpha','beta'),
                PTBart_106=c('psi','xi','gamma','tau','lambda','alpha','beta'),
                PTBart_107=c('psi','xi','tau','lambda','alpha'),
                PTBart_108=c('psi','xi','tau','lambda','alpha','beta'),
                PTBart_109=c('psi','xi','tau','lambda','alpha'),
                PTBart_110=c('psi','xi','tau','lambda','alpha'),
                PTBart_111=c('psi','xi','tau','lambda','alpha','beta'),
                PTBart_112=c('psi','xi','tau','lambda','alpha','beta'),
                PTBart_final_1=c('psi','xi','gamma','tau','lambda'),
                PTBart_final_2=c('psi','xi','gamma','tau','lambda','alpha'),
                PTBart_final_3=c('psi','xi','lambda','tau'),
                PTBart_final_4=c('psi','xi','lambda','tau','alpha'),
                PTBart_final_5=c('psi','xi','lambda','tau'),
                CANDBart_1=c('psi','xi','gamma','tau'),
                CANDBart_2=c('psi','xi','gamma','tau','alpha'),
                CANDBart_3=c('psi','xi','gamma','tau','alpha'),
                STLBart = c('omega_0','vwin','vloss','tau'),
                STLDBart = c('omega_0','vwin','vloss','alpha','tau'),
                bart_par5_5.0 = c('psi','xi','rho','lambda','tau')
  )
  return(params)
}

# Could be manually set if not return with 'Rscript' command in linux
data_type <- args[1]
model_name <- args[2]
data_file_name <- args[3]
df <- read.table(paste('data/',data_type,'/',data_file_name,'.txt',sep=''),header=T)
df <- preprocessing(df)


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

#r_accu = c(0.00, 0.00,0.05, 0.15, 0.25, 0.55, 0.95, 1.45, 2.05, 2.75, 3.45, 4.25, 5.15, 6.00)
r_accu = c(0.00,0.05, 0.15, 0.25, 0.55, 0.95, 1.45, 2.05, 2.75, 3.45, 4.25, 5.15)
r      = c()
for (j in 1:max(L)) {
  r[j] <- r_accu[j+1]-r_accu[j]
}

# Wrap into a list for Stan
dataList <- list(
  N         = n_subj,
  T         = t_max,
  Tsubj     = t_subjs,
  P         = length(r_Accu),#max(pumps) + 1,
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
options(mc.cores = 4)

#nIter     = 2000   # 2000 for test, 4000 for real data
nIter     =4000 
nChains   = 4 
nWarmup   = floor(nIter/2)
nThin     = 1

modelFile = paste('./',model_name,'.stan',sep='')
cat("Estimating", modelFile, "model... \n")
startTime = Sys.time(); print(startTime)
cat("Calling", nChains, "simulations in Stan... \n")
fit = stan(
  modelFile,
  data    = dataList, 
  chains  = nChains,
  iter    = nIter,
  warmup  = nWarmup,
  thin    = nThin,
  #init    = "random",
  # control = list(adapt_delta = 0.999, max_treedepth = 20),
  # control  = list(adapt_delta=0.999, max_treedepth=100),
  #seed    = 233
)
cat("Finishing", modelFile, "model simulation ... \n")
endTime = Sys.time(); print(endTime)  
cat("It took",as.character.Date(endTime - startTime), "\n")

# save the result
save(fit,file=paste('fit_result/',model_name,'_',data_file_name,'.Rdata',sep=''))

param_name <- return_param(model_name)
result_summary<-as.data.frame(rstan::summary(fit,pars=param_name)$summary)
posterior_mean <- extract_posterior(subjs,result_summary)
write.table(posterior_mean,paste('fit_result/summary_',model_name,'_',data_file_name,'.txt',sep=''),quote=F,row.names=F)

