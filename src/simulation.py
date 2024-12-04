#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains 

            # n: inflow
            # s: storage
            # u: release decision
            # r: release
            
            #c: cachuma
            #gi: gibraltar
            #swp: state water proj

"""

import numpy as np
from src.cachuma_lake import Cachuma
from src.gibraltar_lake import Gibraltar
from src.swp_lake import SWP
from src.policy import *
import numpy.matlib as mat
import random
import sys


class log_results:
    pass
    class traj:
        pass
    class cost:
        pass



class SB(object):
   ############# define relevant class parameters
    def __init__(self, opt_par, action_name, capacity, om, cx, t_depl, lifetime):
        self.T           = 12 # period 
        self.gibraltar   = Gibraltar(opt_par.drought_type)
        self.cachuma     = Cachuma(opt_par.drought_type)
        self.swp         = SWP(opt_par.drought_type)
        self.H           = self.gibraltar.H # length of time horizon
        self.Ny          = int(self.H/self.T) #number of years
        self.demand      = np.loadtxt('data/SB_water_demand.txt') 
        self.nom_cost_sw = 100 
        self.nom_cost_rs = 420 

        self.mds   = np.loadtxt('data/Inflow_Individual_Scenarios/mission_pers'+str(opt_par.drought_type[0])+'_sev'+str(opt_par.drought_type[1])+'n_'+str(opt_par.drought_type[2])+'.txt')
        self.sri12 = np.loadtxt('data/Inflow_Individual_Scenarios/gibrSRI12_pers'+str(opt_par.drought_type[0])+'_sev'+str(opt_par.drought_type[1])+'n_'+str(opt_par.drought_type[2])+'.txt')
        self.sri36 = np.loadtxt('data/Inflow_Individual_Scenarios/gibrSRI36_pers'+str(opt_par.drought_type[0])+'_sev'+str(opt_par.drought_type[1])+'n_'+str(opt_par.drought_type[2])+'.txt')
        #self.nsim  = 50 #originally 5, the number of rows in the inflow data you read in, the rows correspond with random scenarios of inflow

        actions = []
        for act in action_name:
            actions.extend(act)
        self.action_name = actions
        self.capacity    = capacity
        self.om          = om
        self.cx          = cx
        self.t_depl      = t_depl
        self.lifetime    = lifetime
        self.portfolio   = []
        self.distr_costs = []
        self.max_swp_market = 275
        self.market_cost  = 1500
        self.curtailment_unitcost = 5998

        H = self.H

        self.opex     = np.zeros(H)
        self.capex    = np.zeros(H)
        self.installed_capacity = np.zeros(H)
        self.reduction_amount = np.zeros(H) #curtailment measures
        self.desal_capac  = np.zeros(H)
        self.wwtp_capac   = np.zeros(H) # waste water treat plant for centralized P and NP reuse
        self.l1_capac     = np.zeros(H) # location of decentralized P and NP
        self.l2_capac     = np.zeros(H)
        self.l3_capac     = np.zeros(H)
        self.l4_capac     = np.zeros(H)
        self.l5_capac     = np.zeros(H)
        self.l6_capac     = np.zeros(H)
        self.l7_capac     = np.zeros(H)

        self.desal_loc    = np.zeros(H)
        self.wwtp_loc     = np.zeros(H)
        self.l1_loc       = np.zeros(H)
        self.l2_loc       = np.zeros(H)
        self.l3_loc       = np.zeros(H)
        self.l4_loc       = np.zeros(H)
        self.l5_loc       = np.zeros(H)
        self.l6_loc       = np.zeros(H)
        self.l7_loc       = np.zeros(H)

        self.def_penalty       = 0
        self.uc_capac          = np.zeros(H)
        self.dis_cost          = 0
        self.surface_cost      = 0
        self.curtailment_cost  = 0
        self.count             = 5
        self.t                 = 0

        self.sc      = [self.cachuma.s0]
        self.sgi     = [self.gibraltar.s0]
        self.sswp    = [self.swp.s0]
        

    def reset(self):

        self.portfolio   = []
        self.distr_costs = []
        self.max_swp_market = 275
        self.market_cost  = 1500
        self.curtailment_unitcost = 5998

        H = self.H

        self.opex     = np.zeros(H)
        self.capex    = np.zeros(H)
        self.installed_capacity = np.zeros(H)
        self.reduction_amount = np.zeros(H) #curtailment measures
        self.desal_capac  = np.zeros(H)
        self.wwtp_capac   = np.zeros(H) # waste water treat plant for centralized P and NP reuse
        self.l1_capac     = np.zeros(H) # location of decentralized P and NP
        self.l2_capac     = np.zeros(H)
        self.l3_capac     = np.zeros(H)
        self.l4_capac     = np.zeros(H)
        self.l5_capac     = np.zeros(H)
        self.l6_capac     = np.zeros(H)
        self.l7_capac     = np.zeros(H)

        self.desal_loc    = np.zeros(H)
        self.wwtp_loc     = np.zeros(H)
        self.l1_loc       = np.zeros(H)
        self.l2_loc       = np.zeros(H)
        self.l3_loc       = np.zeros(H)
        self.l4_loc       = np.zeros(H)
        self.l5_loc       = np.zeros(H)
        self.l6_loc       = np.zeros(H)
        self.l7_loc       = np.zeros(H)

        self.def_penalty       = 0
        self.uc_capac          = np.zeros(H)
        self.dis_cost          = 0
        self.surface_cost      = 0
        self.curtailment_cost  = 0
        self.count             = 5
        self.t                 = 0

        self.sc      = [self.cachuma.s0]
        self.sgi     = [self.gibraltar.s0]
        self.sswp    = [self.swp.s0]

    def simulate(self, a, randseed):
        
        if self.t >= self.H - 1:
            self.reset()

        self.H  = self.gibraltar.H 
        H       = self.H
        self.Ny = H/self.T
        t = self.t

        sys.stdout.write(f"\n{t}\n")

        ncs     = self.cachuma.inflow
        ngis    = self.gibraltar.inflow
        nswps   = self.swp.inflow

        random.seed(randseed)
        s = random.randint(0,len(list(ncs[:,0])))

        nc = []
        ngi = []
        nswp = []
        md = []
        #sri12 = []
        #sri36 = []

        sc = self.sc
        sgi = self.sgi
        sswp = self.sswp

        nc    = list( ncs[s,:] ) 
        ngi   = list( ngis[s,:] )
        nswp  = list( nswps[s,:] )   
        md    = list( self.mds[s,:] ) 
        sri12 = list( self.sri12[s,:] ) 
        sri36 = list( self.sri36[s,:] )

        opex = self.opex
        capex = self.capex
        installed_capacity = self.installed_capacity
        reduction_amount = self.reduction_amount
        desal_capac = self.desal_capac
        wwtp_capac = self.wwtp_capac
        l1_capac = self.l1_capac
        l2_capac = self.l2_capac
        l3_capac = self.l3_capac
        l4_capac = self.l4_capac
        l5_capac = self.l5_capac
        l6_capac = self.l6_capac
        l7_capac = self.l7_capac

        desal_loc = self.desal_loc
        wwtp_loc = self.wwtp_loc
        l1_loc = self.l1_loc
        l2_loc = self.l2_loc
        l3_loc = self.l3_loc
        l4_loc = self.l4_loc
        l5_loc = self.l5_loc
        l6_loc = self.l6_loc
        l7_loc = self.l7_loc

        def_penalty = self.def_penalty
        uc_capac = self.uc_capac
        dis_cost = self.dis_cost
        surface_cost = self.surface_cost
        curtailment_cost = self.curtailment_cost
        count = self.count

        # binary value that indicates whether the plant location is occupied by a plant (1) or not (0)
        Location = {'Desal': 0, 'WWTP': 0, 'L1':0, 'L2':0, 'L3':0, 'L4':0,
                    'L5':0, 'L6':0, 'L7':0}

        # compute value of indicators at time t

        storage_t    = self.compute_stor(sc + sswp + sgi)

        #allocat12t   = self.compute_alloc(t, nc+nswp, 1)
        #allocat36t   = self.compute_alloc(t, nc+nswp, 3)
        #allocat60t   = self.compute_alloc(t, nc+nswp, 5)

        #delta12t     = self.compute_deltas(t, sc, 12) #delta storage over 1 year
        #delta36t     = self.compute_deltas(t, sc, 36)
        #delta60t     = self.compute_deltas(t, sc, 60)

        #sri12t       = sri12[t]
        #sri36t       = sri36[t]

        #installed    = installed_capacity[t]
        #und_constr   = uc_capac[t]
        #curtail_t    = reduction_amount[t]

        #indicators = [storage_t, sri12t, sri36t,
        #                allocat12t, allocat36t, allocat60t,
        #                delta12t, delta36t, delta60t,
        #                installed, und_constr, curtail_t]

        # extract action from policy
        action_decode = []
        for i in ['nothing','SW200','SW300','SW400','SW500','PR200','PR300','PR400','PR500','NPR100']:
            for j in ['nothing','SW200','SW300','SW400','SW500','PR200','PR300','PR400','PR500','NPR100']:
                for k in ['nothing','NPR20','PR50']:
                    for l in ['nothing','NPR20','PR50']:
                        for m in ['nothing','d5','d10','d15','d20']:
                            action_decode.append([i,j,k,l,m])
        policy_cen, policy_rmc, policy_dec, policy_rmd, policy_con = action_decode[a]

        Location['Desal']  = desal_loc[t]
        Location['WWTP']   = wwtp_loc[t]
        Location['L1']     = l1_loc[t]
        Location['L2']     = l2_loc[t]
        Location['L3']     = l3_loc[t]
        Location['L4']     = l4_loc[t]
        Location['L5']     = l5_loc[t]
        Location['L6']     = l6_loc[t]
        Location['L7']     = l7_loc[t]
        #count += 1

        infra_penalty = 0
        # read policy decisions and implement it in model
        # centralized decisions
        if any( [policy_cen=='SW200', policy_cen=='SW300', policy_cen=='SW400', policy_cen=='SW500'] ):
            if Location['Desal'] == 0:
                uc_capac, desal_loc, desal_capac = self.location_track(policy_cen, t, uc_capac, desal_loc, desal_capac)
            else:
                infra_penalty += 10**12
            
        
        if any( [policy_rmc=='SW200', policy_rmc=='SW300', policy_rmc=='SW400', policy_rmc=='SW500'] ):
            if desal_capac[t]>0:
                desal_capac[t+1:H] = 0 #deactivate desal
                desal_loc[t+1:H] = 0
            else:
                infra_penalty += 10**12

        if any( [policy_cen=='PR200', policy_cen=='PR300', policy_cen=='PR400', policy_cen=='PR500', policy_cen=='NPR100'] ):
            if Location['WWTP'] == 0:
                uc_capac, wwtp_loc, wwtp_capac = self.location_track(policy_cen, t, uc_capac, wwtp_loc, wwtp_capac)
            else:
                infra_penalty += 10**12
                
        
        if any( [policy_rmc=='PR200', policy_rmc=='PR300', policy_rmc=='PR400', policy_rmc=='PR500', policy_rmc=='NPR100'] ):
            if wwtp_capac[t]>0:
                wwtp_capac[t+1:H] = 0 #deactivate 
                wwtp_loc[t+1:H] = 0
            else:
                infra_penalty += 10**12
                
        # decentralized decisions        
        if any( [policy_dec=='PR50', policy_dec=='NPR20'] ):
            if Location['L1'] == 0:
                uc_capac, l1_loc, l1_capac = self.location_track(policy_dec, t, uc_capac, l1_loc, l1_capac)
                #count = 0
            elif Location['L2'] == 0:
                uc_capac, l2_loc, l2_capac = self.location_track(policy_dec, t, uc_capac, l2_loc, l2_capac)
                #count = 0
            elif Location['L3'] == 0:
                uc_capac, l3_loc, l3_capac = self.location_track(policy_dec, t, uc_capac, l3_loc, l3_capac)
                #count = 0
            elif Location['L4'] == 0:
                uc_capac, l4_loc, l4_capac = self.location_track(policy_dec, t, uc_capac, l4_loc, l4_capac)
                #count = 0
            elif Location['L5'] == 0:
                uc_capac, l5_loc, l5_capac = self.location_track(policy_dec, t, uc_capac, l5_loc, l5_capac)
                #count = 0
            elif Location['L6'] == 0:
                uc_capac, l6_loc, l6_capac = self.location_track(policy_dec, t, uc_capac, l6_loc, l6_capac)
                #count = 0
            elif Location['L7'] == 0:
                uc_capac, l7_loc, l7_capac = self.location_track(policy_dec, t, uc_capac, l7_loc, l7_capac)
                #count = 0
            else:
                infra_penalty += 10**12

        
        if any( [policy_rmd=='PR50', policy_rmd=='NPR20'] ):
            if l1_loc[t]>0: #at least one distributed plant
                if l7_capac[t]>0:
                    l7_capac[t+1:H] = 0 
                    l7_loc[t+1:H] = 0
                elif l6_capac[t]>0:
                    l6_capac[t+1:H] = 0 
                    l6_loc[t+1:H] = 0
                elif l5_capac[t]>0:
                    l5_capac[t+1:H] = 0 
                    l5_loc[t+1:H] = 0
                elif l4_capac[t]>0:
                    l4_capac[t+1:H] = 0 
                    l4_loc[t+1:H] = 0
                elif l3_capac[t]>0:
                    l3_capac[t+1:H] = 0 
                    l3_loc[t+1:H] = 0
                elif l2_capac[t]>0:
                    l2_capac[t+1:H] = 0  
                    l2_loc[t+1:H] = 0
                elif l1_capac[t]>0:
                    l1_capac[t+1:H] = 0 
                    l1_loc[t+1:H] = 0
            else:
                infra_penalty += -1*10**12
        
        # curtailment decisions
        if any( [policy_con=='d5', policy_con=='d10', policy_con=='d15', policy_con=='d20'] ):
            reduction_amount = self.conservation_measures(t, reduction_amount, policy_con)

        installed_capacity[t] = sum([desal_capac[t], wwtp_capac[t], l1_capac[t], l2_capac[t], l3_capac[t], l4_capac[t], l5_capac[t], l6_capac[t], l7_capac[t]])
            
        
############## simulation of surface water reservoirs
        
        # demand from surface water = total demand - tech installed and curtailment
        dem = self.demand[(t%12)]*( 1 - reduction_amount[t]/100 )
        current_curtail = self.demand[(t%12)]*( reduction_amount[t]/100 )
        d = max( 0, dem - installed_capacity[t] - md[t] )
                
        SS = sc[-1] + sgi[-1] + sswp[-1]
        uc  = sc[-1]/SS 
        ugi = sgi[-1]/SS
        uswp = sswp[-1]/SS
        
        if uswp*d > self.swp.max_release:
            while uswp*d > self.swp.max_release:
                uswp -= 0.05
                uc += 0.04
                ugi += 0.01

        # surface water allocation in cachuma swp and comes in the form of an annual allocation
        # distributed in the month of October for Cachuma and May for SWP
        if (t%12)==9: # October
            nc_ = nc[int((t-9)/self.T)]
        else:
            nc_ = 0
            
        if (t%12)==4: #May
            nswp_ = nswp[int((t-4)/self.T)]
        else:
            nswp_ = 0
            
            
        # mass balance of water reservoirs
        s_, r_c  = self.cachuma.integration(sc[t], uc, nc_, d)
        self.sc.append(s_)

        s_, r_gi  = self.gibraltar.integration(sgi[t], ugi, ngi[t], d)
        self.sgi.append(s_)

        s_, r_swp  = self.swp.integration(sswp[t], uswp, nswp_, d)
        self.sswp.append(s_)
        
        
        # calculation of deficit for penalty
        deficit = max( 0, self.demand[(t%12)] - r_swp - r_c - r_gi - md[t] - installed_capacity[t])
        if deficit < 1e-10:
            deficit = 0
                    
        # restricted purchase of market water to mitigate the deficit 
        max_market = max( 0, self.max_swp_market - r_swp )
        market = min( max_market, deficit ) 
        
        if t>10*12:
            def_penalty += deficit - market
        
############## Calculation of costs
        surface_cost += self.compute_sf_stepcost(r_c, r_gi, md[t], r_swp, market)/10e6
        curtailment_cost += current_curtail*self.curtailment_unitcost/10e6
                
        # distribution costs
        dis_cost += 1.8555*( 1 - reduction_amount[t]/100 )
        if desal_capac[t] > 0:
            dis_cost += 0.240

        if l3_capac[t] == 20:
            dis_cost += - 0.1126
        if l3_capac[t] == 50:
            dis_cost += - 0.1696

        if l6_capac[t] == 20:
            dis_cost += - 0.0149
        if l6_capac[t] == 50:
            dis_cost += - 0.0163

        if l2_capac[t] == 20:
            dis_cost += 0.0121
        if l2_capac[t] == 50:
            dis_cost += - 0.0199

        if l4_capac[t] == 20:
            dis_cost += - 0.0119
        if l4_capac[t] == 50:
            dis_cost += - 0.0127

        if l5_capac[t] == 20:
            dis_cost += - 0.0195
        if l5_capac[t] == 50:
            dis_cost += - 0.0125

        if l7_capac[t] == 20:
            dis_cost += - 0.0096
        if l7_capac[t] == 50:
            dis_cost += - 0.0151

        if l1_capac[t] == 20:
            dis_cost += 0.0014
        if l1_capac[t] == 50:
            dis_cost += - 0.0050
                
        # Technology costs
        capex, opex = self.tech_cost(desal_capac, wwtp_capac, l1_capac, l2_capac, l3_capac, l4_capac, l5_capac, l6_capac, l7_capac)

        # Objective function is total costs + a penalty for deficit
        Jcost = surface_cost/self.Ny + curtailment_cost/self.Ny + opex/self.Ny + capex/self.Ny + dis_cost/self.Ny/10e6 
        Jcost = Jcost + def_penalty + infra_penalty
        #J.append(Jcost)

        self.opex = opex
        self.capex = capex
        self.installed_capacity = installed_capacity
        self.reduction_amount = reduction_amount
        self.desal_capac = desal_capac
        self.wwtp_capac = wwtp_capac
        self.l1_capac = l1_capac
        self.l2_capac = l2_capac
        self.l3_capac = l3_capac
        self.l4_capac = l4_capac
        self.l5_capac = l5_capac
        self.l6_capac = l6_capac
        self.l7_capac = l7_capac

        self.desal_loc = desal_loc
        self.wwtp_loc = wwtp_loc
        self.l1_loc = l1_loc
        self.l2_loc = l2_loc
        self.l3_loc = l3_loc
        self.l4_loc = l4_loc
        self.l5_loc = l5_loc
        self.l6_loc = l6_loc
        self.l7_loc = l7_loc

        self.def_penalty = def_penalty
        self.uc_capac = uc_capac
        #sys.stdout.write(f"{uc_capac}")
        self.dis_cost = dis_cost
        self.surface_cost = surface_cost
        self.curtailment_cost = curtailment_cost
        self.count = count
        self.t = t + 1

        allocat12t   = self.compute_alloc(t+1, nc+nswp, 1)
        allocat36t   = self.compute_alloc(t+1, nc+nswp, 3)
        allocat60t   = self.compute_alloc(t+1, nc+nswp, 5)

        delta12t     = self.compute_deltas(t+1, sc, 12) #delta storage over 1 year
        delta36t     = self.compute_deltas(t+1, sc, 36)
        delta60t     = self.compute_deltas(t+1, sc, 60)

        sri12t       = sri12[t+1]
        sri36t       = sri36[t+1]

        return [Jcost,storage_t, sri12t, sri36t,allocat12t, allocat36t, allocat60t,delta12t, delta36t, delta60t, 
                self.installed_capacity[t+1], self.uc_capac[t+1], self.reduction_amount[t+1]]


    
    
    def planning_policy(self, policy, t, installed_capacity, opex, capex, uc, tech_cap, tech_loc, tech_lifespan):
        i = 0
        H = self.H
        for action in self.action_name:
            if policy == action:
                CXdv = int(self.lifetime[i])
    
                dep   = int(self.t_depl[i])
                T     = t + dep + int(self.lifetime[i])*12
    
                if t+dep < H:
                    y_left = max(0, T-H)
                    T = min(H, T)
                    uc[t:t+dep ]                    = uc[t:t+dep] + float(self.capacity[i])
                    #installed_capacity[t + dep : T] = installed_capacity[t + dep : T] + float(self.capacity[i])
                    #opex[t + dep : T]               = opex[t + dep : T]  + mat.repmat( float(self.om[i])/12, 1, T-dep-t )
                    capex[t + dep : T]              = capex[t + dep : T] + mat.repmat( float(self.cx[i])/CXdv, 1, T-dep-t )
                    tech_cap[t + dep : T]           = tech_cap[t + dep : T] + float(self.capacity[i])
                    tech_loc[t : T]                 = tech_loc[t : T] + 1
                    tech_lifespan[t + dep : T]      = range(int(self.lifetime[i])*12, y_left, -1)
            i += 1
    
        return installed_capacity, opex, capex, uc, tech_cap, tech_loc, tech_lifespan
    
    
    def location_track(self, policy, t, uc, tech_loc, tech_cap):
        i = 0
        H = self.H
        for action in self.action_name:
            
            if policy == action:
                dep   = int(self.t_depl[i])
                if t+dep < H:
                    uc[t:t+dep ]                    = uc[t:t+dep] + float(self.capacity[i])
                    tech_cap[t + dep : H]           = float(self.capacity[i])
                    tech_loc[t : H]                 = 1
            i += 1
    
        return  uc, tech_loc, tech_cap
    
    def cost_from_action(self, capac, act_str):
        t = 1
        H = self.H
        tot_capex = 0
        tot_opex = 0
        
        if sum(capac)>0:
            while t < H:
                if capac[t] > capac[t-1]: #a construction
                    if act_str == 'desal':
                        act_name = 'SW' + str( int(capac[t]) )
                    elif act_str == 'wwtp':
                        if capac[t] == 100:
                            act_name = 'NPR100'
                        else:
                            act_name = 'PR' + str( int(capac[t]) )
                    else:
                        if capac[t] == 20:
                            act_name = 'NPR20'
                        else:
                            act_name = 'PR50'
                    i = 0
                    for action in self.action_name:
                        if act_name == action:
                            capex = float(self.cx[i])
                            opex = float(self.om[i])/12 #O&M per month
                        i+=1
                    tech_life = 0
                    T = t
                    while all([T < self.H-1, capac[T] >= capac[T-1]]) :
                        tech_life += 1
                        T += 1
                    tot_opex += opex*tech_life
                    if all( [t+tech_life >= H, tech_life < 240] ): 
                        tot_capex += capex*(tech_life/480) #reduce end-of-horizon problem
                    else:
                        tot_capex += capex
                    if tech_life > 240:
                        tot_capex += (tech_life - 240)*(capex/240)
                    t += tech_life
                t += 1    
        return tot_capex, tot_opex
    
    def tech_cost(self, desal_capac, wwtp_capac, l1_capac, l2_capac, l3_capac, l4_capac, l5_capac, l6_capac, l7_capac):     
        capex = np.zeros(9)
        opex = np.zeros(9)
        capex[0], opex[0] = self.cost_from_action(desal_capac, 'desal')
        capex[1], opex[1] = self.cost_from_action(wwtp_capac, 'wwtp')
        capex[2], opex[2] = self.cost_from_action(l1_capac, 'dec')
        capex[3], opex[3] = self.cost_from_action(l2_capac, 'dec')
        capex[4], opex[4] = self.cost_from_action(l3_capac, 'dec')
        capex[5], opex[5] = self.cost_from_action(l4_capac, 'dec')
        capex[6], opex[6] = self.cost_from_action(l5_capac, 'dec')
        capex[7], opex[7] = self.cost_from_action(l6_capac, 'dec')
        capex[8], opex[8] = self.cost_from_action(l7_capac, 'dec')
                
                
        return sum(capex), sum(opex)
                    
    def conservation_measures(self, t, reduction_amount, policy):
        if policy == 'd5':
            rr = 5
        elif policy == 'd10':
            rr = 10
        elif policy == 'd15':
            rr = 15
        elif policy == 'd20':
            rr = 20
        else:
            print('unrecognized conservation')
    
    
        Ti = min(self.H, t + 1)
        Tf = min(Ti + 50*12, self.H) #forget effect after 15 years 
        # lognormal distribution
        sigma = 1.03 #shape
        scale = 8.0 #alpha
        reduction_amount[t : Ti]  = [max( exist_red, min(rr, exist_red + rr)) for exist_red in reduction_amount[t : Ti] ]
        surv = [pow(1+pow(tt/12/scale, sigma),-1) for tt in range(Tf - t - 1)]
        reduction_amount[Ti : Tf] = [max( exist_red, min( rr, rr*su))  for exist_red,su in zip(reduction_amount[Ti : Tf], surv) ]
    
        return reduction_amount
    
    def compute_sf_cost(self, rc, rgi, rswp, r_tunnel):
        # surface water
        sw_c = sum( [(c1+c2+c3)*self.nom_cost_sw for c1, c2, c3 in zip(rc, rgi, r_tunnel)] )
        # swp
        swp_c = sum( [ cs * self.nom_cost_rs for cs in rswp ] )
        return sw_c + swp_c
    
    def compute_sf_stepcost(self, rc, rgi, r_tunnel, rswp, market):
        # surface water
        sw_c = (rc+rgi+r_tunnel)*self.nom_cost_sw 
        # swp
        swp_c =  rswp * self.nom_cost_rs 
        # market
        mark_swp = market * self.market_cost 
        return sw_c + swp_c + mark_swp
    
    
    def compute_cost_traj(self, rc, rgi, rswp, r_tunnel):
        # surface water
        sw_c = [(c1+c2+c3)*self.nom_cost_sw for c1, c2, c3 in zip(rc, rgi, r_tunnel)]
        # swp
        swp_c = [ cs * self.nom_cost_rs for cs in rswp ]
    
        csurf = [c1+c2 for c1,c2 in zip(sw_c, swp_c)]
        return csurf
    
    def compute_deltas(self, t, sc, l):
        if t<l:
            delta = 0
        else:
            delta = min( 0, sc[t]  - sc[t-l]) 
        return delta
    
    def compute_alloc(self, t, nc, y):
        curr_y = int(np.floor(t/12)) - 1 #
        y=y-1 
        if (t%12)>=9: #if it's october or later
            curr_y = curr_y + 1
    
        if any( [curr_y == -1, all( [curr_y == y, y == 0] ) ]): #initial months have full allocation
            alloc = 8800
            return alloc
    
        if curr_y<=y:
            prev  = sum( np.ones(y-curr_y)*8800 )
            alloc = ( prev + np.sum(nc[0:curr_y])  )/y
            return alloc
    
        else:
            if y == 0:
                alloc = nc[curr_y]
            else:
                alloc = np.mean(nc[curr_y-y:curr_y])
            return alloc
    
    def compute_stor(self, sc):
        if len(sc) < 12:
            st = np.mean(sc)
        else:
            st = np.mean(sc[-11:])
        return st
