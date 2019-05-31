from statistics import mean
from src.td3.TD3 import TD3
from src.td3.utils import ReplayBuffer
import torch

import gym
import numpy as np
import random
import scipy.stats


class EvolutionaryTD3:
    def __init__(self, n_networks, max_buffer, max_episodes, max_steps, episodes_ready, explore_prob, explore_factors):
        self.n = n_networks  # liczba sieci
        self.max_buffer = max_buffer
        self.max_episodes = max_episodes
        self.max_steps = max_steps

        self.episodes_ready = episodes_ready
        if len(self.episodes_ready) < n_networks:
            print("episodes_ready.len() != n_networks")
            raise Exception

        self.explore_prob = explore_prob - int(explore_prob)
        self.explore_factors = explore_factors

        self.lr = 0.001

        # początkowe ostatnie 10 cząstkowych wyników dla wszystkich sieci ustawiamy na -100
        self.last_ten_scores = [[-100 for _ in range(10)] for _ in range(self.n)]

        self.envs = self.create_envs()
        self.policies = self.create_policies()
        self.replay_buffers = self.create_replay_buffers()



    def create_envs(self):
        envs = []
        for i in range(self.n):
            env = gym.make('BipedalWalker-v2')
            envs.append(env)
        return envs

    def create_policies(self):
        policies = []
        for i in range(self.n):
            state_dim = self.envs[i].observation_space.shape[0]
            action_dim = self.envs[i].action_space.shape[0]
            max_action = float(self.envs[i].action_space.high[0])

            policies.append(TD3(self.lr, state_dim, action_dim, max_action))
        return policies

    def create_replay_buffers(self):
        return [ReplayBuffer() for _ in range(self.n)]

    def multiply_weights(self, idx, factor):
        for param in self.policies[idx].actor.parameters():
            param.data.mul_(factor)
        for param in self.policies[idx].actor_target.parameters():
            param.data.mul_(factor)
        for param in self.policies[idx].critic_1.parameters():
            param.data.mul_(factor)
        for param in self.policies[idx].critic_1_target.parameters():
            param.data.mul_(factor)
        for param in self.policies[idx].critic_2.parameters():
            param.data.mul_(factor)
        for param in self.policies[idx].critic_2_target.parameters():
            param.data.mul_(factor)

    def exploit(self, idx):
        """
        Eksploatacja polega na jednolitym próbkowaniu innego (losowo wybranego) agenta w populacji,
        a następnie porównaniu ostatnich 10 cząstkowych nagród przy użyciu Welch’s t-test.
        Jeśli próbkowany agent ma wyższą średnią cząstkową nagrodę i spełnia warunki t-test,
        wagi z hiperparametrami są kopiowane do obecnego agenta.

        :param idx: indeks sieci, dla której wywołujemy exploit()
        """

        # losujemy indeks sieci różnej od obecnej
        random_idx = random.randrange(self.n)
        while random_idx == idx:
            random_idx = random.randrange(self.n)

        # wybieramy lepszą sieć
        best_net_idx, mean_diff = self.pick_net(idx, random_idx)

        # jeśli wylosowana sieć okazała się być lepsza
        if idx != best_net_idx:
            # kopiujemy tylko część wag z nowej sieci - im większa jest różnica między średnimi cząstkowymi nagrodami
            # dla rozpatrywanych sieci, tym więcej wag zostanie podmienionych (ich ilość zmienia się progowo)
            if 0 <= mean_diff < 10:
                threshold = 0.1
            elif 10 <= mean_diff < 20:
                threshold = 0.2
            elif 20 <= mean_diff < 30:
                threshold = 0.3
            elif 30 <= mean_diff < 40:
                threshold = 0.4
            elif 40 <= mean_diff < 50:
                threshold = 0.5
            else:
                threshold = 0.6

            # podmieniamy wagi
            new_param = self.policies[best_net_idx].actor.parameters()
            for param in self.policies[idx].actor.parameters():
                if random.random() < threshold:
                    param.data.copy_(next(new_param))

            new_param = self.policies[best_net_idx].actor_target.parameters()
            for param in self.policies[idx].actor.parameters():
                if random.random() < threshold:
                    param.data.copy_(next(new_param))

            new_param = self.policies[best_net_idx].critic_1.parameters()
            for param in self.policies[idx].critic_1.parameters():
                if random.random() < threshold:
                    param.data.copy_(next(new_param))

            new_param = self.policies[best_net_idx].critic_1_target.parameters()
            for param in self.policies[idx].critic_1.parameters():
                if random.random() < threshold:
                    param.data.copy_(next(new_param))

            new_param = self.policies[best_net_idx].critic_2.parameters()
            for param in self.policies[idx].actor.parameters():
                if random.random() < threshold:
                    param.data.copy_(next(new_param))

            new_param = self.policies[best_net_idx].critic_2_target.parameters()
            for param in self.policies[idx].actor.parameters():
                if random.random() < threshold:
                    param.data.copy_(next(new_param))

            print("<exploit", idx, "> Wczytano nowe wagi z sieci nr ", best_net_idx)
        else:
            print("<exploit", idx, "> Wagi zostają, obecne są lepsze od sieci nr ", random_idx)

    def explore(self, idx):

        if random.random() < 0.5:
            self.multiply_weights(idx, self.explore_factors[0])
            print("<explore", idx, "> Przemnożono wagi przez ", self.explore_factors[0])
        else:
            self.multiply_weights(idx, self.explore_factors[1])
            print("<explore", idx, "> Przemnożono wagi przez ", self.explore_factors[1])

    def pick_net(self, idx1, idx2):
        """
        Porównanie nagród cząstkowych dwóch sieci przy użyciu Welch's t-test
        :param idx1: obecna sieć
        :param idx2: losowo wybrana sieć
        :return: indeks najlepszej sieci
        """

        statistic, pvalue = scipy.stats.ttest_ind(self.last_ten_scores[idx1], self.last_ten_scores[idx2],
                                                  equal_var=False)
        if pvalue <= 0.05:
            # przeszło welch's t-test, teraz porównanie średnich z ostatnich 10 wyników
            mean_curr = mean(self.last_ten_scores[idx1])
            mean_rand = mean(self.last_ten_scores[idx2])
            mean_diff = abs(mean_curr - mean_rand)
            if mean_curr > mean_rand:
                return idx1, 0  # obecna sieć lepsza
            else:
                return idx2, mean_diff  # losowo wybrana sieć lepsza
        else:
            return idx1, 0  # nie przeszło welch's t-test

    def train(self):
        ######### Hyperparameters #########
        env_name = "BipedalWalker-v2"
        log_interval = 10           # print avg reward after interval
        random_seed = 0
        gamma = 0.99                # discount for future rewards
        batch_size = 100            # num of transitions sampled from replay buffer
        exploration_noise = 0.1
        polyak = 0.995              # target policy update parameter (1-tau)
        policy_noise = 0.2          # target policy smoothing noise
        noise_clip = 0.5
        policy_delay = 2            # delayed policy updates parameter
        max_timesteps = 2000        # max timesteps in one episode
        directory = "./preTrained/{}".format(env_name) # save trained models
        filename = "TD3_{}_{}".format(env_name, random_seed)
        ###################################

        # Liczba iteracji algorytmu
        for episode in range(self.max_episodes):

            for idx in range(self.n):

                # logging variables:
                avg_reward = 0
                ep_reward = 0
                log_f = open("log.txt","w+")

                state = self.envs[idx].reset()
                for t in range(max_timesteps):
                    self.envs[idx].render()
                    # select action and add exploration noise:
                    action = self.policies[idx].select_action(state)
                    action = action + np.random.normal(0, exploration_noise, size=self.envs[idx].action_space.shape[0])
                    action = action.clip(self.envs[idx].action_space.low, self.envs[idx].action_space.high)

                    # take action in env:
                    next_state, reward, done, _ = self.envs[idx].step(action)
                    self.replay_buffers[idx].add((state, action, reward, next_state, float(done)))
                    state = next_state

                    avg_reward += reward
                    ep_reward += reward

                    # if episode is done then update policy:
                    if done or t==(max_timesteps-1):
                        self.policies[idx].update(self.replay_buffers[idx], t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                        break

                # każda sieć ma swój max epizodów, po których zostaną wywołane metody exploit i explore
                if episode % self.episodes_ready[idx] == 0 and episode != 0:
                    self.exploit(idx)
                    if random.random() < self.explore_prob:
                        self.explore(idx)

                # logging updates:
                log_f.write('{},{}\n'.format(episode, ep_reward))
                log_f.flush()
                ep_reward = 0

                # if avg reward > 300 then save and stop traning:
                if (avg_reward/log_interval) >= 300:
                    print("########## Solved! ###########")
                    name = filename + '_solved'
                    self.policies[idx].save(directory, name)
                    log_f.close()
                    break

                if episode > 500:
                    self.policies[idx].save(directory, filename)

                # print avg reward every log interval:
                if episode % log_interval == 0:
                    avg_reward = int(avg_reward / log_interval)
                    print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
                    avg_reward = 0


    def append_score(self, idx, new_score):
        """
        Usuwa ostatni wynik z 10 ostatnich cząstkowych wyników i dodaje nowy
        :param idx: indeks sieci
        :param new_score: nowy wynik
        """
        self.last_ten_scores[idx] = self.last_ten_scores[idx][1:]
        self.last_ten_scores[idx].append(new_score)


