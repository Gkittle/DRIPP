import numpy as np
import random
from src import SB
import pandas as pd

class MDP:
    
    def __init__(self, gamma, state_space, action_space, T, R, TR):
        self.gamma = gamma
        self.state_space = state_space
        self.action_space = action_space
        self.T = T
        self.R = R
        self.TR = TR


class EpsilonGreedyExploration:
    
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, model, s):
        
        A = model.action_space
        if random.random() < self.epsilon:
            return random.choice(A)
        else:
            return np.argmax([model.lookahead(s, a) for a in A])


class QLearning:
    
    def __init__(self, state_space, action_space, gamma, Q, alpha):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.Q = Q
        self.alpha = alpha

    def lookahead(self, s, a):
        return self.Q[s, a]

    def update(self, s, a, r, s_prime):
        max_next_q = np.max(self.Q[s_prime, :])
        self.Q[s, a] += self.alpha * (r + self.gamma * max_next_q - self.Q[s, a])


def simulate(P, model, policy, h, s):
    for _ in range(h):
        # Select action using the policy
        a = policy(model, s)
        # Sample transition and reward
        s_prime, r = P.TR(s, a)
        # Update the model
        model.update(s, a, r, s_prime)
        # Move to the next state
        s = s_prime

def state_numeration(ss,sri1,sri3,al1,al3,al5,del1,del3,del5,in_cap,un_cap,curt):
    
    int1 = 35*6*6*11*11*11*40*40*40*8*8*6
    return

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
model = SB(opt_par, action_name, capacity, om, cx, t_depl, lifetime)


# Initialize MDP and parameters
state_space = range(35*6*6*11*11*11*40*40*40*8*8*6)
action_space = range(4500)
gamma = 0.9

R = lambda s, a, randseed: model.simulate(s, a, randseed)
TR = lambda s, a, randseed: (T(s,a,randseed), R[s,a,randseed])

P = MDP(gamma, state_space, action_space, T, R, TR)

# Initialize Q-learning model
Q = np.zeros((len(state_space), len(action_space)))
alpha = 0.2
model = QLearning(state_space, action_space, gamma, Q, alpha)

# Initialize policy
epsilon = 0.1
policy = EpsilonGreedyExploration(epsilon)

# Simulate
k = 20  # Number of steps

initial_state = 0

simulate(P, model, policy, k, initial_state)
