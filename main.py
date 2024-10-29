#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script. 
The following script runs a quick demo optimization for one seed and few function 
iterations. This is not intended to reproduce the results of the paper, but 
simply to demonstrate the methods. 
The optimization parameters used in the full-scale experiments of the 
paper are included in this script and can be run upon changing the binary condition in 
line 113. 

"""


import sys
sys.path.append('src')
sys.path.append('ptreeopt')
from src import SB, SBsim
from src import *
from ptreeopt import PTreeOpt, MultiprocessingExecutor
import logging
import csv
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd

# set optimization parameters
class OptimizationParameters(object):
    def __init__(self):
        self.cores    = 36 # this value is used only in the full scale optimization
        self.nseeds   = 1
        self.nobjs    = 1
        self.drought_type = [87, 0.83, 2] # set drought type here [Persistence (months), Intensity (unitless), Frequency (droughts/100 years)]

#create Result class
class Result():
    pass

#initialize action table
action = [] #Action Name
capacity = [] #Capacity (Acre-Feet/Month)
om = [] #Operation and Maintenance Costs (Million Dollars/ Acre-Feet/ Year)
cx = [] #CAPEX (Million Dollars)
lifetime = [] #Years
t_depl = [] #Time to Deployment (Month)
action_type = [] # types 1: Deploy Centralized Infrastructure, 2: Commision Decentralized Infrastructure, 3: Decommission Centralized Infrastructure, 4: Decommission Centralized Infrastructure, 5: Conservation

#Read the action table
df_actions_table = pd.read_csv("data/actions_table.csv")
action = df_actions_table['action'].values
capacity = df_actions_table['capacity (af/month)'].values
om = df_actions_table['OPEX ($10^6/af/year)'].values
cx = df_actions_table['CAPEX($10^6)'].values
t_depl = df_actions_table['time to deployment (month)'].values
lifetime = df_actions_table['lifetime (year)'].values
action_type = [int(val) for val in df_actions_table['action_type'].values]

action_name = [[], [], [], [], []]
i = 0
for act in action:
        action_name[action_type[i]].extend( [act] )
        i += 1

opt_par = OptimizationParameters()

# define parameters for model and algorithm 
model = SB(opt_par, action_name, capacity, om, cx, t_depl, lifetime) 
algorithm = PTreeOpt(model.simulate,
                     feature_bounds=[[0, 35000],
                                     [-3, 3], [-3, 3],
                                     [0, 12100],[0, 12100],[0, 12100],
                                     [-20000, 20000],[-20000, 20000],[-20000, 20000],
                                     [0,800], [0, 800], [0, 25] ],

                     feature_names=['Surface Storage',
                                    'SRI 1y', 'SRI 3y',
                                    'allocation 1y','allocation 3y','allocation 5y',
                                    'delta storage 1y','delta storage 3y','delta storage 5y',
                                    'installed capacity', 'capacity under construction', 'curtailment'],

                     discrete_actions=True,
                     action_names=action_name,
                     mu=10, 
                     cx_prob=0.70,
                     population_size=10, #set this parameter to 100 for full scale optimization and to 10 for scaled down 
                     max_depth=3,
                     multiobj=False,
                     )

if __name__ == '__main__':
    
    seed = 1 # initializing random seed
    
##### uncomment the following for random seed initialization on computing cluster #####
    #parser = argparse.ArgumentParser()
    #parser.add_argument("process", help="seed number")
    #args = parser.parse_args()
    #seed = int(args.process)
    #np.random.rand(seed)


    logging.basicConfig(level=logging.INFO,
        format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')

    opt_par = OptimizationParameters()


##### set following condition to 1 for full scale optimization on computing cluster
    if 0:
        with MultiprocessingExecutor(processes=opt_par.cores) as executor:
            best_solution, best_score, snapshots = algorithm.run(max_nfe=300000, #max_nfe in full scale is 300,000
                                                         log_frequency=100,
                                                         snapshot_frequency=100,
                                                         executor=executor,
                                                         drought_type = opt_par.drought_type,
                                                         seed = seed)
    else:
        best_solution, best_score, snapshots = algorithm.run(max_nfe=20, #parameters for short optimization
                                                         log_frequency=10,
                                                         snapshot_frequency=10,
                                                         drought_type = opt_par.drought_type)


    result = Result()
    result.best_solution = best_solution
    result.best_score = best_score
    result.snapshots = snapshots
    result.model = model
    string = 'results/test_results' + str(opt_par.drought_type[0]) + '_' + str(opt_par.drought_type[1]) + '_' + str(seed) +'.dat'

    with open(string, 'wb') as f: 
        pickle.dump(result, f)
        
        
####### simulate best_result to visualize some trajectories
    model_sim = SBsim(opt_par, action_name, capacity, om, cx, t_depl, lifetime) 
    log = model_sim.simulate(best_solution, 0)
    
####### some demo plots
    #plt.style.use('seaborn-darkgrid')

    fig, axs = plt.subplots(3)
    fig.suptitle('For demo purpose only \n These results are not converged')
    
    axs[0].fill_between(range(1200), log.sri36, where = (np.array(log.sri36)>0), color = '#73A5C6') 
    axs[0].set_ylabel('SRI [-]')
    axs[0].fill_between(range(1200), log.sri36, where = (np.array(log.sri36)<0), color = '#ff0000' )
    
    axs[1].fill_between(range(1200), log.capacity, color = '#FF6600' )
    axs[1].set_ylabel('Capacity \n [AF/month]')
    
    axs[2].fill_between(range(1200), log.sc, color = '#00316E' )
    axs[2].set_xlabel('Time [months]')
    axs[2].set_ylabel('Storage [AF]')

    fig.savefig("plot_example.png")
    
    
    
    
    
    
    
    