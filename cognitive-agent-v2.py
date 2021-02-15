#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# objective is to teach the agent which slot must be choose in order to avoid interference
import gym
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ns3gym import ns3env

env = gym.make('ns3-v0', port=5555)
print("environment",env)

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 4000
SHOW_EVERY = 3000

ob_space = env.observation_space #4
ac_space = env.action_space #4
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.n)

s_size = ob_space.shape[0]
a_size = ac_space.n


epsilon = 1.0               # exploration rate
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

STATS_EVERY = 100
# For Stats
ep_rewards = []
aggr_ep_rewards ={'ep':[], 'avg':[], 'max':[], 'min':[]}

q_table = np.random.randint(low = 0, high = 4, size = [s_size]+[a_size])

def get_channel_position(a):
    
    if a == [1,0,0,0] or a==None or a==[] or a==[0,0,0,0]:
        ch_pos=0
    elif a == [0,1,0,0] or a==[1,1,0,0]:
        ch_pos=1
    elif a == [0,0,1,0]:
        ch_pos=2
    elif a == [0,0,0,1]:
        ch_pos=3
    print(ch_pos)
    return ch_pos

for episode in range(EPISODES):
    
    episode_reward = 0
    
    state = env.reset()
    print("stato", state)
    channel_position = get_channel_position(state)
    print("channel position",channel_position)
    if state == None or state== []:
        state = [0,0,0,0]
    state = np.reshape(state, [1, s_size])
    done = False

    if episode % SHOW_EVERY == 0:
        render=True
        print (episode)
    else:
        render=False

    while not done:


        # Choose action
        if np.random.rand(1) < epsilon:
            # get random action
            action = np.random.randint(a_size)
            print("azione casuale", action)
        else:
            # get action from Q table
            action = np.argmax(q_table[channel_position])
            print("azione predetta", action)

        # Step
        next_state, reward, done, _ = env.step(action)
        print("nuovo stato", next_state)
        new_channel_position = get_channel_position(next_state)
        if next_state == None or next_state== []:
            next_state = [0,0,0,0]
        next_state = np.reshape(next_state, [1, s_size])
        
        episode_reward += reward
        
        if episode % SHOW_EVERY ==0:
            env.render()
        #new_q = (1 - LEARNING_RATE)* current_q + LEARNING_RATE*(reward+DISCOUNT*max_future_q)
        
        # if siulation did not end yet after last step - update Q table
        if  not done:
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_channel_position])
            
            #Current Q value (for current state and performed action
            current_q = q_table[channel_position,action]

            # And here's our equation for a new Q value for current state action
            new_q = (1 - LEARNING_RATE)* current_q + LEARNING_RATE*(reward+DISCOUNT*max_future_q)

            # update Q table with a new Q value
            q_table[channel_position,action]=new_q
        channel_position=new_channel_position

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode>=START_EPSILON_DECAYING: 
        epsilon -= epsilon_decay_value
    

    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        print(f'Episode:{episode:>5}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')


    np.save(f"qtables/{episode}-qtable.npy", q_table)
# AT THE END
#if episode % 10 ==0:
#    np.save(f"qtables/{episode}-qtable.npy", q_table)


env.close()

print("Plot Learning Performance")
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(10,4))
plt.grid(True, linestyle='--')
plt.title('Learning Performance')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
#plt.plot(range(len(time_history)), time_history, label='Steps', marker="^", linestyle=":")#, color='red')
#plt.plot(range(len(rew_history)), rew_history, label='Reward', marker="", linestyle="-")#, color='k')
#plt.xlabel('Episode')
#plt.ylabel('Time')
plt.legend(prop={'size': 12})

plt.savefig('learning.pdf', bbox_inches='tight')
plt.show()
