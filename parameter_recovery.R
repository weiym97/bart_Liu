rm(list=ls())
posterior = read.table('fit_real_data/Posterior_distribution.txt',header=T)

#posterior_group_1 = posterior[posterior$Group =='1',]
#posterior_group_2 = posterior[posterior$Group =='2',]
#posterior_group_3 = posterior[posterior$Group =='3',]


#t.test(posterior_group_1$rho,posterior_group_2$rho)
#t.test(posterior_group_1$rho,posterior_group_3$rho)
#t.test(posterior_group_2$rho,posterior_group_3$rho)

#t.test(posterior_group_1$lambda,posterior_group_2$lambda)
#t.test(posterior_group_1$lambda,posterior_group_3$lambda)
#t.test(posterior_group_2$lambda,posterior_group_3$lambda)

#posterior_psi_group_1 = coef(fitdist(posterior_group_1$psi,distr="gamma",method="mle"))
#posterior_xi_group_1 = coef(fitdist(posterior_group_1$xi,distr="gamma",method="mle"))
#posterior_rho_group_1 = coef(fitdist(posterior_group_1$rho,distr="gamma",method="mle"))
#posterior_lambda_group_1 = coef(fitdist(posterior_group_1$lambda,distr="gamma",method="mle"))
#posterior_tau_group_1 = coef(fitdist(posterior_group_1$tau,distr="gamma",method="mle"))

#posterior_psi_group_2 = coef(fitdist(posterior_group_2$psi,distr="gamma",method="mle"))
#posterior_xi_group_2 = coef(fitdist(posterior_group_2$xi,distr="gamma",method="mle"))
#posterior_rho_group_2 = coef(fitdist(posterior_group_2$rho,distr="gamma",method="mle"))
#posterior_lambda_group_2 = coef(fitdist(posterior_group_2$lambda,distr="gamma",method="mle"))
#posterior_tau_group_2 = coef(fitdist(posterior_group_2$tau,distr="gamma",method="mle"))

#posterior_psi_group_3 = coef(fitdist(posterior_group_3$psi,distr="gamma",method="mle"))
#posterior_xi_group_3 = coef(fitdist(posterior_group_3$xi,distr="gamma",method="mle"))
#posterior_rho_group_3 = coef(fitdist(posterior_group_3$rho,distr="gamma",method="mle"))
#posterior_lambda_group_3 = coef(fitdist(posterior_group_3$lambda,distr="gamma",method="mle"))
#posterior_tau_group_3 = coef(fitdist(posterior_group_3$tau,distr="gamma",method="mle"))

#posterior_group_result = data.frame(params=c('psiShape','psiRate','xiShape','xiRate',
#                                             'rhoShape','rhoRate','lambdaShape','lambdaRate','tauShape','tauRate'),
#                                    group1=c(posterior_psi_group_1,posterior_xi_group_1,posterior_rho_group_1,
#                                             posterior_lambda_group_1,posterior_tau_group_1),
#                                    group2=c(posterior_psi_group_2,posterior_xi_group_2,posterior_rho_group_2,
#                                             posterior_lambda_group_2,posterior_tau_group_2),
#                                    group3=c(posterior_psi_group_3,posterior_xi_group_3,posterior_rho_group_3,
#                                             posterior_lambda_group_3,posterior_tau_group_3))

#write.table(x=posterior_group_result,file="ParameterRecoveryLiu/posterior_group_result.txt",
#            sep=",",na="??",quote=FALSE,row.names=FALSE)

posterior_psi = coef(fitdist(posterior$psi,distr="gamma",method="mle"))
posterior_xi = coef(fitdist(posterior$xi,distr="gamma",method="mle"))
posterior_rho = coef(fitdist(posterior$rho,distr="gamma",method="mle"))
posterior_lambda = coef(fitdist(posterior$lambda,distr="gamma",method="mle"))
posterior_tau = coef(fitdist(posterior$tau,distr="gamma",method="mle"))

posterior_group_result_all = data.frame(params=c('psiShape','psiRate','xiShape','xiRate',
                                             'rhoShape','rhoRate','lambdaShape','lambdaRate','tauShape','tauRate'),
                                    group=c(posterior_psi,posterior_xi,posterior_rho,
                                            posterior_lambda,posterior_tau))

write.table(x=posterior_group_result,file="fit_real_data/posterior_group_result_all.txt",
                        sep=",",na="??",quote=FALSE,row.names=FALSE)


