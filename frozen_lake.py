import gym
import random
import numpy as np

env = gym.make('FrozenLake-v0')

print(env.observation_space.n)

def value_iteration(env, gamma = 0.4):

    value_table = np.zeros(env.observation_space.n)
    num_iterations = 100000
    threshold = 1e-20

    for i in range(num_iterations):

        updated_value_table = np.copy(value_table)

        for state in range(env.observation_space.n):

            Q_value = []

            for action in range(env.action_space.n):

                next_states_rewards = []

                for next_sr in env.P[state][action]:

                    trans_prob, next_state, reward_prob, _ = next_sr
                    next_states_rewards.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state])))

                Q_value.append(np.sum(next_states_rewards))
                value_table[state] = max(Q_value)

                if(np.sum(np.fabs(updated_value_table - value_table)) <= threshold):

                    print("Value iteration converged at iteration %d." %(i + 1))
                    break
    return value_table, Q_value

                
            


