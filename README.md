# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.'

## States:
The environment has 7 states:

Two Terminal States: G: The goal state & H: A hole state. Five Transition states / Non-terminal States including S: The starting state.

## Actions:
The agent can take two actions:

R: Move right. L: Move left.

## Transition Probabilities:
The transition probabilities for each action are as follows:

50% chance that the agent moves in the intended direction. 33.33% chance that the agent stays in its current state. 16.66% chance that the agent moves in the opposite direction. For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

## Rewards:
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

## POLICY EVALUATION FUNCTION
~~~



import warnings ; warnings.filterwarnings('ignore')
!pip install gym_walk

import gym, gym_walk
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123)
# Reference https://github.com/mimoralea/gym-walk 

pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk

def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)

def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)

env = gym.make('SlipperyWalkFive-v0')
P = env.env.P
init_state = env.reset()
goal_state = 6
LEFT, RIGHT = range(2)
P

init_state

state, reward, done, info = env.step(RIGHT)
print("state:{0} - reward:{1} - done:{2} - info:{3}".format(state, reward, done, info))

# First Policy
pi_1 = lambda s: {
    0:LEFT, 1:LEFT, 2:LEFT, 3:LEFT, 4:LEFT, 5:LEFT, 6:LEFT
}[s]
print_policy(pi_1, P, action_symbols=('<', '>'), n_cols=7)

# Find the probability of success and the mean return of the first policy
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_1, goal_state=goal_state)*100,
    mean_return(env, pi_1)))

# Create your own policy
pi_2 = lambda s: {
    0:RIGHT, 1:RIGHT, 2:RIGHT, 3:RIGHT, 4:RIGHT, 5:RIGHT, 6:RIGHT
}[s]
# Write your code here

print_policy(pi_2, P, action_symbols=('<', '>'), n_cols=7)

# Find the probability of success and the mean return of you your policy

#write your code here

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_2, goal_state=goal_state)*100,
    mean_return(env, pi_2)))

# Compare your policy with the first policy
if((probability_success(env, pi_1, goal_state=goal_state)*100,mean_return(env, pi_1))>(probability_success(env, pi_2, goal_state=goal_state)*100,mean_return(env, pi_2))):
  print('The FIst policy is better than the Second one...')
else:
  print('The Second policy is the better one...')

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    num_states = len(P)
    V = np.zeros(num_states, dtype=np.float64)
    while True:
        delta = 0
        for s in range(num_states):
            v = V[s]
            new_v = 0
            a = pi(s)  
            for prob, next_state, reward, done in P[s][a]:
                new_v += prob * (reward + gamma * V[next_state])
            V[s] = new_v
            delta = max(delta, abs(v - V[s]))   
        if delta < theta:
            break
    return V

# Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P,gamma=0.99)
print_state_value_function(V1, P, n_cols=7, prec=5)

# Code to evaluate the second policy
# Write your code here
V2 = policy_evaluation(pi_2, P,gamma=0.99)
print_state_value_function(V2, P, n_cols=7, prec=5)

if np.max(V1) > np.max(V2):
    print("Policy 1 (pi_1) is better based on the maximum state value.")
else:
    print("Policy 2 (pi_2) is better based on the maximum state value.")

V1
print_state_value_function(V1, P, n_cols=7, prec=5)
V2
print_state_value_function(V2, P, n_cols=7, prec=5)
V1>=V2

if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
~~~
## OUTPUT:
## Policy 1:
![image](https://github.com/user-attachments/assets/912b80ab-da31-43ee-ab87-ec7e04ec5e6d)
![image](https://github.com/user-attachments/assets/5d319146-ce01-46e5-9f03-3d4c16b53478)
![image](https://github.com/user-attachments/assets/8e8fbeaa-eba5-43f0-b36c-47d5b68afd7d)
## Policy 2:
![image](https://github.com/user-attachments/assets/fe2504dd-d7d2-4b00-be4d-3fcf8fb586b8)
![image](https://github.com/user-attachments/assets/117540e9-8142-48f4-9643-c5914335c93e)
![image](https://github.com/user-attachments/assets/54d47a46-af25-434e-b412-e5c6c39ab47f)


## Comparision:
![image](https://github.com/user-attachments/assets/e34f983d-4f8d-49c7-9ef7-cae41bc98945)

## Conclusion:
![image](https://github.com/user-attachments/assets/fcdb54e0-7f34-4bc7-8a62-7c0472b7b27f)

## RESULT:

Thus, a Python program is developed to evaluate the given policy.
