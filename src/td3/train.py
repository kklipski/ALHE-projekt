import torch
import gym
import numpy as np
from src.td3.TD3 import TD3
from src.td3.utils import ReplayBuffer

def train():
    ######### Hyperparameters #########
    env_name = "BipedalWalker-v2"
    log_interval = 1           # print avg reward after interval
    random_seed = 0
    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    exploration_noise = 0.1 
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = 1000         # max num of episodes
    max_timesteps = 2000        # max timesteps in one episode
    directory = "./preTrained/" # save trained models
    filename = "TD3_{}_{}".format(env_name, random_seed)
    stop_condition = 100
    ###################################
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # logging variables:
    avg_reward = 0
    ep_reward = 0
    log_f = open("log.txt","w+")
    
    # training procedure:
    for episode in range(1, max_episodes+1):
        state = env.reset()
        no_reward_counter = -100  # jeśli dojdzie do 100 uznajemy, że robot się nie porusza
        start_no_reward_counter = 0
        for t in range(max_timesteps):
            env.render()
            # select action and add exploration noise:
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
            action = action.clip(env.action_space.low, env.action_space.high)
            
            # take action in env:
            next_state, reward, done, _ = env.step(action)

            # jeśli robot stoi w miejscu przez stop_condition kroków, zakończ pętlę
            if no_reward_counter == 0:
                start_no_reward_counter = reward
                # print("START=",start_no_reward_counter)
            if start_no_reward_counter - 0.05 < reward < start_no_reward_counter + 0.05:
                # print("COUNTER=",no_reward_counter)
                no_reward_counter += 1
            else:
                no_reward_counter = 0

            # jeśli robot stoi w miejscu dłuższy czas, ustaw flagę
            stand_flag = no_reward_counter > stop_condition

            if stand_flag:
                stuck_pen_flag = True
                if ep_reward <= 0:
                    reward = reward + -100
                    stuck_pen = -100
                elif 0 < ep_reward <= 50:
                    reward = reward + -49
                    stuck_pen = -49
                elif 50 < ep_reward <= 100:
                    reward = reward + -36
                    stuck_pen = -36
                elif 100 < ep_reward <= 150:
                    reward = reward + -25
                    stuck_pen = -25
                elif 150 < ep_reward <= 200:
                    reward = reward + -16
                    stuck_pen = -16
                elif 200 < ep_reward <= 250:
                    reward = reward + -9
                    stuck_pen = -9
                else:
                    reward = reward + -4
                    stuck_pen = -4

            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            
            avg_reward += reward
            ep_reward += reward
            
            # if episode is done then update policy:
            if done or t == (max_timesteps-1):
                policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                break
        
        # logging updates:
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()
        ep_reward = 0
        
        # if avg reward > 300 then save and stop traning:
        if (avg_reward/log_interval) >= 300:
            print("########## Solved! ###########")
            name = filename + '_solved'
            policy.save(directory, name)
            log_f.close()
            break

        if episode % 100 == 0 and episode != 0:
            policy.save(directory, filename)

        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = int(avg_reward / log_interval)
            print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            avg_reward = 0


if __name__ == '__main__':
    train()
