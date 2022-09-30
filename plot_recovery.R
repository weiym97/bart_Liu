rm(list=ls())
args <- commandArgs(trailingOnly = TRUE)

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

model_name <- args[1]
data_name <- args[2]
outlier <- args[3]

stat_sim <- read.csv(paste('data/simulation/',data_name,'_statistics.csv',sep=''))
stat_recov <- read.table(paste('fit_result/summary_',model_name,'_',data_name,'.txt',sep=''),header=T)

stat_sim
stat_recov

if (outlier=='remove'){
  index < - stat_recov$tau <=200
  stat_sim < - stat_sim[index,]
  stat_recov <- stat_recov[index,]
}

params=return_param(model_name)
for (i in 1:length(params)){
  jpeg(paste(file_name='plot_result/',model_name,'_',data_name,'_',params[i],'.jpg',sep=''))
  plot(stat_sim[,params[i]],stat_recov[,params[i]],xlab='True',ylab='Recovery',main=paste(params[i],'correlation=',cor(stat_sim[,params[i]],stat_recov[,params[i]])))
  dev.off()
  }

