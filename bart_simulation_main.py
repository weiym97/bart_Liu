import time

import numpy as np
import pandas as pd

from bart_model import *


def bart_generate_basic_info(config, save_dir):
    save_dir = save_dir + config['trial_id'] + '/'
    result = pd.read_excel(save_dir + 'result.xlsx')
    result.drop(['trial', 'pumps', 'explosion'], axis=1, inplace=True)
    result.drop_duplicates(inplace=True)
    result.to_excel(save_dir + 'basic_info.xlsx', index=False)


def bart_plot_stop_main(config, save_dir):
    save_dir = save_dir + config['trial_id'] + '/'
    result = pd.read_excel(save_dir + 'result.xlsx')
    plot_dir = save_dir + 'plot_stop/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    max_subjID = int(max(result['SubjID']))

    for subjID in range(1, max_subjID + 1):
        subresult = result[result['SubjID'] == subjID]
        subresult = subresult[subresult['explosion'] == 0]
        plt.hist(subresult['pumps'], range=(0, config['max_pump'] - 1))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=20)
        plt.xlabel('Pump Number', fontsize=20)
        plt.title('SubjectID ' + str(subjID), fontsize=25)
        plt.savefig(plot_dir + 'subjID=' + str(subjID) + '.jpg')
        plt.close()


def bart_plot_pump_prob(config, save_dir):
    save_dir = save_dir + config['trial_id'] + '/'
    result = pd.read_excel(save_dir + 'result.xlsx')
    plot_dir = save_dir + 'plot_pump_prob/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    max_subjID = int(max(result['SubjID']))

    for subjID in range(1, max_subjID + 1):
        subresult = result[result['SubjID'] == subjID]
        subresult_succeed = subresult[subresult['explosion'] == 0]
        subresult_fail = subresult[subresult['explosion'] == 1]
        pumps_succeed = np.array(subresult_succeed['pumps'])
        pumps_fail = np.array(subresult_fail['pumps'])

        pumps = np.zeros(config['max_pump'] - 1)
        no_pumps = np.zeros(config['max_pump'] - 1)
        for i in range(config['max_pump'] - 1):
            no_pumps[i] = np.sum(pumps_succeed == i + 1)
            pumps[i] = np.sum(pumps_succeed > i + 1) + np.sum(pumps_fail == i + 1)
        pump_prob = pumps / (pumps + no_pumps + 1e-8)
        plt.plot(np.arange(1, config['max_pump']), pump_prob)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=20)
        plt.xlabel('Pump Number', fontsize=20)
        plt.title('SubjectID ' + str(subjID) + ' : pump probability', fontsize=25)

        plt.savefig(plot_dir + 'subjID=' + str(subjID) + '.jpg')
        plt.close()

class EWBartOld():
    def __init__(self, max_pump, explode_prob, accu_reward, num_trial=50,):
        self.max_pump = max_pump
        self.explode_prob = explode_prob
        self.accu_reward = accu_reward
        self.reward = np.diff(accu_reward)
        self.num_trial = num_trial

        self.burst = (np.repeat(np.reshape(self.explode_prob,(1,-1)),self.num_trial,axis=0) > np.random.uniform(size = (self.num_trial,1)))

    def generate_data(self, psi, xi, rho, Lambda, tau):
        pumps = np.zeros(self.num_trial)
        explode = np.zeros(self.num_trial)

        n_success = 0
        n_pumps = 0

        for i in range(self.num_trial):
            burst = self.burst[i,:]
            n_pumps_use = n_pumps
            n_success_use = n_success

            if n_pumps_use > 0:
                p_temp = (n_pumps_use - n_success_use) / n_pumps_use
            else:
                p_temp = 0

            p_burst = np.exp(-xi * n_pumps_use) * psi + (1 - np.exp(-xi * n_pumps_use)) * p_temp

            for j in range(self.max_pump):
                utility = (1 - p_burst) * self.reward[j] ** rho - p_burst * Lambda * (
                        self.accu_reward[j]) ** rho
                p_pump = 1 / (1 + np.exp(-tau * utility))

                pump = int(np.random.binomial(1, p_pump, 1)) > 0
                if not pump:
                    pumps[i] = j
                    explode[i] = 0
                    break
                elif burst[j]:
                    pumps[i] = j + 1
                    explode[i] = 1
                    break

            n_success += pumps[i] - explode[i]
            n_pumps += pumps[i]
        return pumps.astype(np.int32), explode.astype(np.int32)


def model_simulation_main(model_name,accu_reward, explode_prob, max_pump, params, data_dir,
                          n_simu_subj=1000, n_fit_per_run=100, trial_per_subj=50,group='1'):
    params.to_excel(data_dir + model_name + '_group_' + group +'_Nsubj_' + str(n_simu_subj) + '_simulation_statistics.xlsx', index=False)
    params.to_csv(data_dir + model_name + '_group_' + group + '_Nsubj_' + str(n_simu_subj) + '_simulation_statistics.csv', index=False)
    model = EWBartOld(max_pump=max_pump, accu_reward=accu_reward, explode_prob=explode_prob, num_trial=trial_per_subj)

    result = []
    for i in range(n_simu_subj):
        print('###########################')
        print(params['subjID'][i])
        pumps,explosion = model.generate_data(psi=params['psi'][i],
                                                  xi=params['xi'][i],
                                                  rho=params['rho'][i],
                                                  Lambda=params['lambda'][i],
                                                  tau=params['tau'][i])

        subjdata = pd.DataFrame({'subjID': params['subjID'][i],
                                 'group': 0,
                                 'trial': np.arange(trial_per_subj) + 1,
                                 'reward': accu_reward[pumps],
                                 'pumps': pumps,
                                 'explosion': explosion})
        result.append(subjdata)
    result = pd.concat(result)
    result.to_csv(data_dir + model_name + '_group_' + group +'_Nsubj_' + str(n_simu_subj) + '_simulation.txt', sep=' ', index=False, doublequote=False)
    #for j in range(n_file):
    #    result_save = result.iloc[int(j * n_fit_per_run * trial_per_subj):int((j + 1) * n_fit_per_run * trial_per_subj),:]
    #    result_save.to_csv(data_dir + model_name +'_simulation_'+str(j+1)+'.txt',sep=' ',index=False,doublequote=False)

if __name__ == '__main__':
    accu_reward = np.array([0.0, 0.05, 0.15, 0.25, 0.55, 0.95, 1.45, 2.05, 2.75, 3.45, 4.25, 5.15, 6.0])
    explode_prob = np.array([0.021, 0.042, 0.063, 0.146, 0.239, 0.313, 0.438, 0.563, 0.688, 0.792, 0.896, 1.0])
    max_pump = 12

    # Totally, we simulation n_simu_subj subjects, but we cut them into n_fit_per_run to run parallelly
    n_simu_subj = 100
    n_fit_per_run = 100

    ###############################################################################################################
    # Simulation for Liu's paper
    '''
    parameter_group_result = pd.read_csv('fit_real_data/posterior_group_result.txt')
    #print(parameter_group_result)
    #print(parameter_group_result.iloc[0])

    ### Group 1
    psi = np.random.gamma(shape=parameter_group_result.iloc[0,1],scale=1/parameter_group_result.iloc[1,1],size=n_simu_subj)
    xi = np.random.gamma(shape=parameter_group_result.iloc[2,1],scale=1/parameter_group_result.iloc[3,1],size=n_simu_subj)
    rho = np.random.gamma(shape=parameter_group_result.iloc[4,1],scale=1/parameter_group_result.iloc[5,1],size=n_simu_subj)
    Lambda = np.random.gamma(shape=parameter_group_result.iloc[6,1],scale=1/parameter_group_result.iloc[7,1],size=n_simu_subj)
    tau = np.random.gamma(shape=parameter_group_result.iloc[8,1],scale=1/parameter_group_result.iloc[9,1],size=n_simu_subj)

    #psi = np.clip(psi,0.0,0.05)
    psi = np.clip(psi,0.0,1.0)
    #rho = np.clip(rho,0.15,2.0)
    rho = np.clip(rho,0.0,2.0)
    #Lambda = np.clip(Lambda,0,18)
    #tau = np.clip(tau,0,35)

    params = pd.DataFrame({'subjID': np.arange(n_simu_subj) + 10001,
                           'psi': psi,
                           'xi': xi,
                           'rho': rho,
                           'lambda':Lambda,
                           'tau': tau
                           })

    data_dir = 'data/simulation/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    model_simulation_main('EWBart',accu_reward, explode_prob, max_pump, params, data_dir, n_simu_subj, n_fit_per_run,group='1')

    ### Group 2
    psi = np.random.gamma(shape=parameter_group_result.iloc[0,2],scale=1/parameter_group_result.iloc[1,2],size=n_simu_subj)
    xi = np.random.gamma(shape=parameter_group_result.iloc[2,2],scale=1/parameter_group_result.iloc[3,2],size=n_simu_subj)
    rho = np.random.gamma(shape=parameter_group_result.iloc[4,2],scale=1/parameter_group_result.iloc[5,2],size=n_simu_subj)
    Lambda = np.random.gamma(shape=parameter_group_result.iloc[6,2],scale=1/parameter_group_result.iloc[7,2],size=n_simu_subj)
    tau = np.random.gamma(shape=parameter_group_result.iloc[8,2],scale=1/parameter_group_result.iloc[9,2],size=n_simu_subj)

    #psi = np.clip(psi,0.0,0.05)
    psi = np.clip(psi,0.0,1.0)
    #rho = np.clip(rho,0.15,2.0)
    rho = np.clip(rho,0.0,2.0)
    #Lambda = np.clip(Lambda,0,18)
    #tau = np.clip(tau,0,35)

    params = pd.DataFrame({'subjID': np.arange(n_simu_subj) + 20001,
                           'psi': psi,
                           'xi': xi,
                           'rho': rho,
                           'lambda':Lambda,
                           'tau': tau
                           })

    data_dir = 'data/simulation/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    model_simulation_main('EWBart',accu_reward, explode_prob, max_pump, params, data_dir, n_simu_subj, n_fit_per_run,group='2')

    ### Group 3
    psi = np.random.gamma(shape=parameter_group_result.iloc[0,3],scale=1/parameter_group_result.iloc[1,3],size=n_simu_subj)
    xi = np.random.gamma(shape=parameter_group_result.iloc[2,3],scale=1/parameter_group_result.iloc[3,3],size=n_simu_subj)
    rho = np.random.gamma(shape=parameter_group_result.iloc[4,3],scale=1/parameter_group_result.iloc[5,3],size=n_simu_subj)
    Lambda = np.random.gamma(shape=parameter_group_result.iloc[6,3],scale=1/parameter_group_result.iloc[7,3],size=n_simu_subj)
    tau = np.random.gamma(shape=parameter_group_result.iloc[8,3],scale=1/parameter_group_result.iloc[9,3],size=n_simu_subj)

    #psi = np.clip(psi,0.0,0.05)
    psi = np.clip(psi,0.0,1.0)
    #rho = np.clip(rho,0.15,2.0)
    rho = np.clip(rho,0.0,2.0)
    #Lambda = np.clip(Lambda,0,18)
    #tau = np.clip(tau,0,35)

    params = pd.DataFrame({'subjID': np.arange(n_simu_subj) + 30001,
                           'psi': psi,
                           'xi': xi,
                           'rho': rho,
                           'lambda':Lambda,
                           'tau': tau
                           })

    data_dir = 'data/simulation/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    model_simulation_main('EWBart',accu_reward, explode_prob, max_pump, params, data_dir, n_simu_subj, n_fit_per_run,group='3')
    '''
    ### Group 4: We take all the participants as a single group, and try to figure out their distribution
    '''
    param_group_result_all = parameter_group_result = pd.read_csv('fit_real_data/posterior_group_result_all.txt')
    psi = np.random.gamma(shape=parameter_group_result.iloc[0,1],scale=1/parameter_group_result.iloc[1,1],size=n_simu_subj)
    xi = np.random.gamma(shape=parameter_group_result.iloc[2,1],scale=1/parameter_group_result.iloc[3,1],size=n_simu_subj)
    rho = np.random.gamma(shape=parameter_group_result.iloc[4,1],scale=1/parameter_group_result.iloc[5,1],size=n_simu_subj)
    Lambda = np.random.gamma(shape=parameter_group_result.iloc[6,1],scale=1/parameter_group_result.iloc[7,1],size=n_simu_subj)
    tau = np.random.gamma(shape=parameter_group_result.iloc[8,1],scale=1/parameter_group_result.iloc[9,1],size=n_simu_subj)

    #psi = np.clip(psi,0.0,0.05)
    psi = np.clip(psi,0.0,1.0)
    #rho = np.clip(rho,0.15,2.0)
    rho = np.clip(rho,0.0,2.0)
    #Lambda = np.clip(Lambda,0,18)
    #tau = np.clip(tau,0,35)

    params = pd.DataFrame({'subjID': np.arange(n_simu_subj) + 40001,
                           'psi': psi,
                           'xi': xi,
                           'rho': rho,
                           'lambda':Lambda,
                           'tau': tau
                           })

    data_dir = 'data/simulation/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    model_simulation_main('EWBart',accu_reward, explode_prob, max_pump, params, data_dir, n_simu_subj, n_fit_per_run,group='4')
    '''

    '''
    params = pd.read_csv('fit_real_data/Posterior_distribution.csv')
    params['subjID'] = params['ID']
    data_dir = 'data/simulation/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    model_simulation_main('EWBart',accu_reward, explode_prob, max_pump, params, data_dir,len(params), len(params),group='5')
    '''

    posterior_real_data = pd.read_excel('fit_real_data/恢复性检验参数.xlsx')
    print(posterior_real_data)

    posterior_real_data_group_6 = posterior_real_data.loc[(posterior_real_data['Group']==1),:]
    posterior_real_data_group_7 = posterior_real_data.loc[(posterior_real_data['Group']==2),:]
    posterior_real_data_group_8 = posterior_real_data.loc[(posterior_real_data['Group']==3),:]

    ### Group 6
    psi = np.mean(posterior_real_data_group_6['psi'])
    xi = np.mean(posterior_real_data_group_6['xi'])
    rho = np.mean(posterior_real_data_group_6['rho'])
    Lambda = np.mean(posterior_real_data_group_6['lambda'])
    tau = np.mean(posterior_real_data_group_6['tau'])

    params = pd.DataFrame({'subjID': np.arange(n_simu_subj) + 60001,
                           'psi': np.ones(n_simu_subj) * psi,
                           'xi': np.ones(n_simu_subj) * xi,
                           'rho': np.ones(n_simu_subj) * rho,
                           'lambda':np.ones(n_simu_subj) * Lambda,
                           'tau': np.ones(n_simu_subj) * tau,
                           })

    data_dir = 'data/simulation/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    model_simulation_main('EWBart',accu_reward, explode_prob, max_pump, params, data_dir, n_simu_subj, n_fit_per_run,group='6')

    ### Group 7

    psi = np.mean(posterior_real_data_group_7['psi'])
    xi = np.mean(posterior_real_data_group_7['xi'])
    rho = np.mean(posterior_real_data_group_7['rho'])
    Lambda = np.mean(posterior_real_data_group_7['lambda'])
    tau = np.mean(posterior_real_data_group_7['tau'])

    params = pd.DataFrame({'subjID': np.arange(n_simu_subj) + 70001,
                           'psi': np.ones(n_simu_subj) * psi,
                           'xi': np.ones(n_simu_subj) * xi,
                           'rho': np.ones(n_simu_subj) * rho,
                           'lambda':np.ones(n_simu_subj) * Lambda,
                           'tau': np.ones(n_simu_subj) * tau,
                           })

    data_dir = 'data/simulation/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    model_simulation_main('EWBart',accu_reward, explode_prob, max_pump, params, data_dir, n_simu_subj, n_fit_per_run,group='7')


    ### Group 8
    psi = np.mean(posterior_real_data_group_8['psi'])
    xi = np.mean(posterior_real_data_group_8['xi'])
    rho = np.mean(posterior_real_data_group_8['rho'])
    Lambda = np.mean(posterior_real_data_group_8['lambda'])
    tau = np.mean(posterior_real_data_group_8['tau'])

    params = pd.DataFrame({'subjID': np.arange(n_simu_subj) + 80001,
                           'psi': np.ones(n_simu_subj) * psi,
                           'xi': np.ones(n_simu_subj) * xi,
                           'rho': np.ones(n_simu_subj) * rho,
                           'lambda':np.ones(n_simu_subj) * Lambda,
                           'tau': np.ones(n_simu_subj) * tau,
                           })

    data_dir = 'data/simulation/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    model_simulation_main('EWBart',accu_reward, explode_prob, max_pump, params, data_dir, n_simu_subj, n_fit_per_run,group='8')




    #print(posterior_real_data['rho'] == '1')


    '''
    data_dir = 'data/simulation/'
    params = pd.DataFrame({'subjID': ['0000'],
                           'psi': [0.0315],
                           'xi': [0.000329],
                           'rho': [0.1849],
                           'lambda':[28.96],
                           'tau': [45.8459]
                           })
    model_simulation_main('EWBart',accu_reward, explode_prob, max_pump, params, data_dir, n_simu_subj, n_fit_per_run)
    '''