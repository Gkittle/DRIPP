import numpy as np
import random
from src import SB
import pandas as pd
from src.cachuma_lake import Cachuma
from src.gibraltar_lake import Gibraltar
from src.swp_lake import SWP
import sys
import tempfile
import time

class MDP:
    
    def __init__(self, gamma, state_space, action_space, TR):
        self.gamma = gamma
        self.state_space = state_space
        self.action_space = action_space
        #self.T = T
        #self.R = R
        self.TR = TR

def compute_alloc(t, nc, y):
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
        
def compute_stor(sc):
        if len(sc) < 12:
            st = np.mean(sc)
        else:
            st = np.mean(sc[-11:])
        return st

def compute_deltas(t, sc, l):
        if t<l:
            delta = 0
        else:
            delta = min( 0, sc[t]  - sc[t-l]) 
        return delta


class EpsilonGreedyExploration:
    
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, model, s):
        random.seed(int(1000*time.time())% 2**32)
        A = model.action_space
        r = random.random()
        #sys.stdout.write(f"{r}")
        if r < self.epsilon:
            return random.choice(A)
        else:
            for _ in range(1199):
                a_i = 0
                for val in Q_mask:
                    if val[0] == s:
                        a_n = val[1]
                        if Q[int(s),int(a_n)] >= Q[int(s), int(a_i)]:
                            a_i = a_n
            return a_i


class QLearning:
    
    def __init__(self, state_space, action_space, gamma, Q, Q_mask, alpha):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.Q = Q
        self.Q_mask = Q_mask
        self.alpha = alpha

    def lookahead(self, s, a):
        #sys.stdout.write(f"{s}")
        #sys.stdout.write(f"{a}")
        return self.Q[int(s), int(a)]

    def update(self, s, a, r, s_prime):
        max_next_q = np.max(self.Q[int(s_prime), :])
        self.Q_mask.append([int(s),int(a)])
        self.Q[int(s), int(a)] += self.alpha * (r + self.gamma * max_next_q - self.Q[int(s), int(a)])


def simulate(P, model, policy, h, s, randseed):
    cost = 0
    reward = 0
    for val in range(h):
        #sys.stdout.write(f"\n{val}\n")
        # Select action using the policy
        a = policy(model, s)
        # Sample transition and reward
        #s_prime, r = P.TR(s, a)
        sim_out = P.TR(a,randseed)
        r = sim_out[0]
        s_prime = state_numeration(sim_out[2:])
        # Update the model
        model.update(s, a, r, s_prime)
        # Move to the next state
        s = s_prime
        cost += sim_out[1]
        reward += r
    return reward,cost

def state_numeration(rounded):
    #rounded = [ss,sri1,sri3,al1,al3,al5,del1,del3,del5,in_cap,un_cap,curt,desal_bool,wwtp_bool,decen_bool]
    #sys.stdout.write(f"\n{rounded}\n")
    step1 = 3500
    step2 = 2
    step3 = 1100
    step4 = 10000
    step5 = 100
    step6 = 5
    for val,i in zip(np.arange(0,35001,step1), range(int(35000/step1))):
        if (rounded[0] >= val):
            if(rounded[0] <= val + step1):
                rounded[0] = i
    for val,i in zip(np.arange(-3,3.1,step2), range(int(6/step2))):
        if (rounded[1] >= val):
            if(rounded[1] <= val + step2):
                rounded[1] = i
        if (rounded[2] >= val):
            if(rounded[2] <= val + step2):
                rounded[2] = i
    for val,i in zip(np.arange(0,12100,step3), range(int(12100/step3))):
        if (rounded[3] >= val):
            if(rounded[3] <= val + step3):
                rounded[3] = i
        if (rounded[4] >= val):
            if(rounded[4] <= val + step3):
                rounded[4] = i
        if (rounded[5] >= val):
            if(rounded[5] <= val + step3):
                rounded[5] = i
    for val,i in zip(np.arange(-20000,20001,step4),range(int(40000/step4))):
        if (rounded[6] >= val):
            if(rounded[6] <= val + step4):
                rounded[6] = i
        if (rounded[7] >= val):
            if(rounded[7] <= val + step4):
                rounded[7] = i
        if (rounded[8] >= val):
            if(rounded[8] <= val + step4):
                rounded[8] = i
    for val,i in zip(np.arange(0,801,step5),range(int(800/step5))):
        if (rounded[9] >= val):
            if(rounded[9] <= val + step5):
                rounded[9] = i
        if (rounded[10] >= val):
            if(rounded[10] <= val + step5):
                rounded[10] = i
    for val,i in zip(np.arange(0,26,step6),range(int(25/step6))):
        if (rounded[11] >= val):
            if(rounded[11] <= val + step6):
                rounded[11] = i
    """
    int1 = (35000/step1)*((6/step2)**2)*((12100/step3)**3)*((40000/step4)**3)*((800/step5)**2)*(25/step6)
    int2 = (35000/step1)*((6/step2)**2)*((12100/step3)**3)*((40000/step4)**3)*((800/step5)**2)
    int3 = (35000/step1)*((6/step2)**2)*((12100/step3)**3)*((40000/step4)**3)*((800/step5)**1)
    int4 = (35000/step1)*((6/step2)**2)*((12100/step3)**3)*((40000/step4)**3)
    int5 = (35000/step1)*((6/step2)**2)*((12100/step3)**3)*((40000/step4)**2)
    int6 = (35000/step1)*((6/step2)**2)*((12100/step3)**3)*((40000/step4)**1)
    int7 = (35000/step1)*((6/step2)**2)*((12100/step3)**3)
    int8 = (35000/step1)*((6/step2)**2)*((12100/step3)**2)
    int9 = (35000/step1)*((6/step2)**2)*((12100/step3)**1)
    int10 = (35000/step1)*((6/step2)**2)
    int11 = (35000/step1)*((6/step2)**1)
    int12 = (35000/step1)
    """
    #num = (int1*rounded[0] + int2*rounded[1] + int3*rounded[2] + int4*rounded[3] + int5*rounded[4] + int6*rounded[5] + 
    #       int7*rounded[6] + int8*rounded[7] + int9*rounded[8] + int10*rounded[9] + int11*rounded[10] + int12*rounded[11])
    """
    int1 = (35000/step1)*((6/step2)**1)*((12100/step3)**1)*((40000/step4)**1)*((800/step5)**2)*(25/step6)
    int2 = (35000/step1)*((6/step2)**1)*((12100/step3)**1)*((40000/step4)**1)*((800/step5)**2)
    int3 = (35000/step1)*((6/step2)**1)*((12100/step3)**1)*((40000/step4)**1)*((800/step5)**1)
    int4 = (35000/step1)*((6/step2)**1)*((12100/step3)**1)*((40000/step4)**1)
    int5 = (35000/step1)*((6/step2)**1)*((12100/step3)**1)
    int6 = (35000/step1)*((6/step2)**1)
    int7 = (35000/step1)
    """
    #int1 = (2**3)*(35000/step1)*((12100/step3)**1)*((40000/step4)**1)*((800/step5)**2)*(25/step6)
    #int3 = (2**3)*(35000/step1)*((12100/step3)**1)*((40000/step4)**1)*((800/step5)**2)
    #int4 = (2**3)*(35000/step1)*((12100/step3)**1)*((40000/step4)**1)*((800/step5)**1)
    #int5 = (2**3)*(35000/step1)*((12100/step3)**1)*((40000/step4)**1)
    #int6 = (2**3)*(35000/step1)*((12100/step3)**1)
    #int7 = (2**3)*(35000/step1)
    #int8 = 2**3
    #int9 = 2**2
    #int10 = 2
    #num = (int1*rounded[0] + int3*rounded[3] + int4*rounded[6] + int5*rounded[9] + int6*rounded[10] + int7*rounded[11] + int8*rounded[12]
    #       + int9*rounded[13] + int10*rounded[14])
    #sys.stdout.write(f"\n{rounded}\n")
    #sys.stdout.write(f"{num}\n")
    int1 = 4*6*2*2*2
    int2 = 6*2*2*2
    int3 = 2*2*2
    int4 = 2*2
    int5 = 2
    num = (int1*(rounded[3] - 1) + int2*(rounded[6] - 1) + int3*(rounded[11] - 1) + int4*(rounded[12] - 1) + int5*(rounded[13] - 1) + rounded[14])
    return num

# set optimization parameters
class OptimizationParameters(object):
    def __init__(self):
        self.cores    = 36 # this value is used only in the full scale optimization
        self.nseeds   = 1
        self.nobjs    = 1
        self.drought_type = [87, 0.83, 2] # set drought type here [Persistence (months), Intensity (unitless), Frequency (droughts/100 years)]

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
modelSB = SB(opt_par, action_name, capacity, om, cx, t_depl, lifetime)


# Initialize MDP and parameters
#state_space = range(35*6*6*11*11*11*40*40*40*8*8*6)
#state_space = range(35*11*40*16*16*6*2*2*2)
state_space = range(11*4*6*2*2*2)
action_space = range(4500)
gamma = 0.9

Sim = lambda a, randseed: modelSB.simulate(a, randseed) 
#return [Jcost, Jcost1, ss, sri12t, sri36t,allocat12t, allocat36t, allocat60t,delta12t, delta36t, delta60t, installed_capacity, uc_capac, reduction_amount]

P = MDP(gamma, state_space, action_space, Sim)

# Initialize Q-learning model
Q = np.memmap(tempfile.NamedTemporaryFile().name, dtype='float32',mode='w+',shape=(len(state_space),len(action_space)))
#Q = np.memmap(tempfile.NamedTemporaryFile().name, dtype='float32',mode='w+',shape=((350*6*110*400*80*80*6),len(action_space)))
Q_mask = []
Q_infeasible = []
alpha = 0.2
model = QLearning(state_space, action_space, gamma, Q, Q_mask, alpha)

# Initialize policy
epsilon = 0.7
policy = EpsilonGreedyExploration(epsilon)

# Simulate
k = 2*1198  # Number of steps

gibraltar   = Gibraltar(opt_par.drought_type)
cachuma     = Cachuma(opt_par.drought_type)
swp         = SWP(opt_par.drought_type)
mds   = np.loadtxt('data/Inflow_Individual_Scenarios/mission_pers'+str(opt_par.drought_type[0])+'_sev'+str(opt_par.drought_type[1])+'n_'+str(opt_par.drought_type[2])+'.txt')
sri12 = np.loadtxt('data/Inflow_Individual_Scenarios/gibrSRI12_pers'+str(opt_par.drought_type[0])+'_sev'+str(opt_par.drought_type[1])+'n_'+str(opt_par.drought_type[2])+'.txt')
sri36 = np.loadtxt('data/Inflow_Individual_Scenarios/gibrSRI36_pers'+str(opt_par.drought_type[0])+'_sev'+str(opt_par.drought_type[1])+'n_'+str(opt_par.drought_type[2])+'.txt') 

ncs     = cachuma.inflow
ngis    = gibraltar.inflow
nswps   = swp.inflow

randseed = 0
random.seed(randseed)
s = random.randint(0,len(list(ncs[:,0])))

nc    = list( ncs[s,:] ) 
ngi   = list( ngis[s,:] )
nswp  = list( nswps[s,:] )   
md    = list( mds[s,:] ) 
sri12 = list( sri12[s,:] ) 
sri36 = list( sri36[s,:] )
sc    = [cachuma.s0]
sgi     = [gibraltar.s0]
sswp    = [swp.s0]

storage_t    = compute_stor(sc + sswp + sgi)

allocat12t   = compute_alloc(0, nc+nswp, 1)
allocat36t   = compute_alloc(0, nc+nswp, 3)
allocat60t   = compute_alloc(0, nc+nswp, 5)

delta12t     = compute_deltas(0, sc, 12) #delta storage over 1 year
delta36t     = compute_deltas(0, sc, 36)
delta60t     = compute_deltas(0, sc, 60)

sri12t       = sri12[0]
sri36t       = sri36[0]

initial_state = state_numeration([storage_t, sri12t, sri36t, allocat12t, allocat36t, allocat60t, delta12t, delta36t, delta60t, 0, 0, 0, 0, 0, 0])
#sys.stdout.write(f"\n{[storage_t, sri12t, sri36t, allocat12t, allocat36t, allocat60t, delta12t, delta36t, delta60t, 0, 0, 0]}\n")
reward, cost = simulate(P, model, policy, k, initial_state, randseed)
sys.stdout.write(f"{reward}")
sys.stdout.write(f"{cost}")

final_actions = []
final_states = []
s = initial_state
final_states.append(s)
final_reward = 0
final_cost = 0
for _ in range(1199):
    a_i = 540 #random state
    for val in Q_mask:
        if val[0] == s:
            a_n = val[1]
            if Q[int(s),int(a_n)] >= Q[int(s), int(a_i)]:
                a_i = a_n

    a = a_i
    sim_out = P.TR(a,randseed)
    r = sim_out[0]
    final_reward += r
    final_cost += sim_out[1]
    s_prime = state_numeration(sim_out[2:])
    s = s_prime
    action_decode = []
    for i in ['nothing','SW200','SW300','SW400','SW500','PR200','PR300','PR400','PR500','NPR100']:
        for j in ['nothing','SW200','SW300','SW400','SW500','PR200','PR300','PR400','PR500','NPR100']:
            for k in ['nothing','NPR20','PR50']:
                for l in ['nothing','NPR20','PR50']:
                    for m in ['nothing','d5','d10','d15','d20']:
                        action_decode.append([i,j,k,l,m])
    policy_cen, policy_rmc, policy_dec, policy_rmd, policy_con = action_decode[a]
    final_actions.append([policy_cen, policy_rmc, policy_dec, policy_rmd, policy_con])
    final_states.append(s)
sys.stdout.write("\nFinal\n")
sys.stdout.write(f"\n{final_reward}\n")
sys.stdout.write(f"\n{final_cost}\n")
d = {'states': final_states[:-1], 'actions': final_actions}
df = pd.DataFrame(data=d)
df.to_csv('Results.csv', index = False)