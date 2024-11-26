import numpy as np
import pandas as pd
import sys
import time
import random

start_time = time.time()

class MDP:
    def __init__(self, gamma, state_space, action_space):
        self.gamma = gamma  # Discount factor
        self.state_space = state_space  # State space
        self.action_space = action_space  # Action space

def epsilon_greedy_exploration(model, s, epsilon):

    A = model.action_space  # Action space
    if random.random() < epsilon:
        return random.choice(A)
    
    def Q(s, a):
        return model.lookahead(s, a)
    
    return max(A, key=lambda a: Q(s, a))

class QLearning:
    def __init__(self, state_space, action_space, gamma, Q, alpha):
        self.state_space = state_space  # State space
        self.action_space = action_space  # Action space
        self.gamma = gamma  # Discount factor
        self.Q = Q  # Action-value function
        self.alpha = alpha  # Learning rate

    def update(self, s, a, r, s_prime):
        self.Q[s, a] += self.alpha * (r + self.gamma * np.max(self.Q[s_prime, :]) - self.Q[s, a])
        return self
    
    
def simulate(P, model, policy, h, s):

    for _ in range(h):
        # Determine action using the policy
        a = policy(model, s)
        # Apply the action and get the next state and reward
        s_prime, r = P.TR(s, a)
        # Update the model with the new experience
        model.update(s, a, r, s_prime)
        # Move to the next state
        s = s_prime



# Initialization of Q-learning parameters
alpha = 0.1 # Learning rate

P = MDP(states, actions, 0.9)
Q = np.zeros((len(states), len(actions)))
model = QLearning(P.state_space, P.action_space, P.gamma, Q, alpha)
policy = epsilon_greedy_exploration(0.1)
h = 20 #number of steps to simulate
s = 1 #initial state
simulate(P, model, policy, h, s)

Q_final = model.Q
pi_final = np.argmax(Q_final, axis = 1)