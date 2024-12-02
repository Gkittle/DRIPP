import numpy as np
import random

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


# Initialize MDP and parameters
state_space = range(10)  # Example state space
action_space = range(4500)
gamma = 0.9  # Discount factor
T = lambda s, a: (s + a) % len(state_space)  # Example transition function
R = lambda s, a: -abs(s - a)  # Example reward function
TR = lambda s, a: (T(s, a), R(s, a))  # Combined transition-reward function

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
